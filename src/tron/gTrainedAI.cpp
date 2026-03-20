/*

*************************************************************************

ArmageTron -- Just another Tron Lightcycle Game in 3D.
Copyright (C) 2026

**************************************************************************

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

***************************************************************************

*/

#include "gTrainedAI.h"

#include "eGrid.h"
#include "gSensor.h"
#include "tConfiguration.h"
#include "tConsole.h"
#include "tDirectories.h"
#include "tRandom.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace
{
enum
{
    kBaseFeatures = 44,
    kHistoryFrames = 250,
    kCurrentFrameOffset = ( kHistoryFrames - 1 ) * kBaseFeatures,
    kFeatures = kBaseFeatures * kHistoryFrames,
    kTemporalHidden = 4096,
    kHidden1 = 1024,
    kHidden2 = 512,
    kHidden3 = 256,
    kActions = 3
};

enum
{
    kParameterCount =
        kFeatures * kTemporalHidden + kTemporalHidden +
        kTemporalHidden * kHidden1 + kHidden1 +
        kHidden1 * kHidden2 + kHidden2 +
        kHidden2 * kHidden3 + kHidden3 +
        kActions * kHidden3 + kActions +
        kHidden3 + 1
};

enum FeatureIndex
{
    kBias = 0,
    kFront,
    kFrontNarrowLeft,
    kFrontNarrowRight,
    kFrontLeft,
    kFrontRight,
    kWideLeft,
    kWideRight,
    kLeft,
    kRight,
    kBackLeft,
    kBackRight,
    kBack,
    kSpeed,
    kCanLeft,
    kCanRight,
    kTurnDelay,
    kEnemyAhead,
    kEnemySide,
    kEnemyNear,
    kEnemySpeedDiff,
    kEnemyHeadingDot,
    kEnemyCrossing,
    kEnemyFrontProximity,
    kEnemyBearingDrift,
    kFrontEnemyWall,
    kFrontRimWall,
    kLeftEnemyWall,
    kRightEnemyWall,
    kLeftRimWall,
    kRightRimWall,
    kLeftRightBalance,
    kFrontPressure,
    kBackPressure,
    kWallCrowding,
    kForwardArcSafety,
    kSideArcSafety,
    kEnemyClosing,
    kEnemyLateralClosing,
    kEnemyBackProximity,
    kSpeedPressure,
    kEscapeLeft,
    kEscapeRight,
    kEscapeRouteBias
};

static bool sg_enable = false;
// Keep live training opt-in; recording can stay on.
static bool sg_learn = false;
static bool sg_record = true;
static bool sg_autostart = false;
static bool sg_offlineTrain = false;
// -1 means every AI player uses Blacklight. Positive values limit how many
// bot players are controlled by Blacklight when classic bots are also present.
static int sg_botCount = -1;
static char const * const sg_aiName = "Blacklight";
static tString sg_modelFile( "trained_ai_model.txt" );
static tString sg_recordFile( "trained_ai_experience.log" );
static tString sg_metricsFile( "trained_ai_metrics.csv" );
static tString sg_checkpointPrefix( "trained_ai_checkpoints/blacklight" );
static tString sg_offlineSourceList( "" );
static tString sg_offlineStateFile( "trained_ai_offline_state.txt" );

static REAL sg_thinkTime = .12f;
static REAL sg_lookAheadSeconds = 2.5f;
static REAL sg_exploration = .10f;

static REAL sg_learningRate = .02f;
static REAL sg_valueLearningRate = .02f;
static REAL sg_baselineDecay = .02f;
static REAL sg_discount = .985f;
static REAL sg_weightDecay = .0002f;
static REAL sg_weightClip = 5.0f;

static int sg_saveEvery = 25;
static int sg_maxEpisodeSteps = 512;
static int sg_checkpointEvery = 50;
static int sg_trainEpochs = 2;
static int sg_recordStride = 1;

// self-play variance: periodically snapshot the current policy and sample
// opponents from that snapshot pool.
static int sg_policyPoolSize = 8;
static int sg_policySnapshotEvery = 40;
static int sg_policySnapshotWarmup = 80;
static REAL sg_policyHistoricProb = .35f;

static REAL sg_rewardDistance = .002f;
static REAL sg_rewardWin = 2.0f;
static REAL sg_rewardDeath = -2.0f;

static REAL Clamp( REAL value, REAL low, REAL high )
{
    if ( value < low ) return low;
    if ( value > high ) return high;
    return value;
}

static REAL Max3( REAL a, REAL b, REAL c )
{
    REAL best = a;
    if ( b > best ) best = b;
    if ( c > best ) best = c;
    return best;
}

static REAL Max4( REAL a, REAL b, REAL c, REAL d )
{
    REAL best = a;
    if ( b > best ) best = b;
    if ( c > best ) best = c;
    if ( d > best ) best = d;
    return best;
}

static REAL RandomUnit()
{
    return tRandomizer::GetInstance().Get();
}

static REAL RandomSigned()
{
    return ( RandomUnit() - .5f ) * 2.0f;
}

static int ActionToTurn( int action )
{
    if ( action <= 0 ) return -1;
    if ( action >= 2 ) return 1;
    return 0;
}

struct Step
{
    REAL x[kFeatures];
    REAL h0[kTemporalHidden];
    REAL h1[kHidden1];
    REAL h2[kHidden2];
    REAL h3[kHidden3];
    REAL p[kActions];
    REAL v;
    int action;
    bool canLeft;
    bool canRight;
};

static unsigned long long sg_recordEpisodeId = 0;

static void RecordEpisode( std::vector< Step > const & episode, bool survived, REAL distance, REAL reward )
{
    if ( !sg_record || episode.empty() )
    {
        return;
    }

    std::ofstream out;
    if ( !tDirectories::Var().Open( out, static_cast< char const * >( sg_recordFile ), std::ios::app ) )
    {
        return;
    }

    out.setf( std::ios::fixed );
    out.precision( 6 );
    ++sg_recordEpisodeId;

    int stride = sg_recordStride < 1 ? 1 : sg_recordStride;

    for ( unsigned int i = 0; i < static_cast< unsigned int >( episode.size() ); ++i )
    {
        if ( i % static_cast< unsigned int >( stride ) != 0 )
        {
            continue;
        }
        Step const & step = episode[i];
        out << sg_recordEpisodeId << " " << i << " " << ( survived ? 1 : 0 ) << " " << distance << " " << reward << " " << step.action;
        out << " " << ( step.canLeft ? 1 : 0 ) << " " << ( step.canRight ? 1 : 0 );
        for ( int a = 0; a < kActions; ++a )
        {
            out << " " << step.p[a];
        }
        out << " " << step.v;
        for ( int f = 0; f < kFeatures; ++f )
        {
            out << " " << step.x[f];
        }
        out << "\n";
    }
}

static REAL AveragePredictedValue( std::vector< Step > const & episode )
{
    if ( episode.empty() )
    {
        return 0;
    }

    REAL total = 0;
    for ( std::vector< Step >::const_iterator it = episode.begin(); it != episode.end(); ++it )
    {
        total += it->v;
    }

    return total / static_cast< REAL >( episode.size() );
}

class TrainingMetrics
{
public:
    static TrainingMetrics & Get()
    {
        static TrainingMetrics metrics;
        return metrics;
    }

    void RecordEpisode( bool survived, REAL distance, REAL reward, REAL averagePredictedValue, unsigned int steps, bool learned, unsigned int policyEpisodes, unsigned int policyUpdates )
    {
        char const * metricsFile = static_cast< char const * >( sg_metricsFile );
        if ( !metricsFile || !metricsFile[0] )
        {
            return;
        }

        ++episodes_;
        if ( survived )
        {
            ++wins_;
        }
        rewardTotal_ += reward;
        distanceTotal_ += distance;
        predictedValueTotal_ += averagePredictedValue;
        stepsTotal_ += steps;

        bool writeHeader = NeedHeader();

        std::ofstream out;
        if ( !tDirectories::Var().Open( out, metricsFile, std::ios::app ) )
        {
            return;
        }

        out.setf( std::ios::fixed );
        out.precision( 6 );

        if ( writeHeader )
        {
            out << "episode,survived,reward,distance,average_predicted_value,steps,learned,policy_episodes,policy_updates,cumulative_win_rate,cumulative_average_reward,cumulative_average_distance,cumulative_average_predicted_value\n";
            headerNeeded_ = false;
        }

        out << episodes_
            << "," << ( survived ? 1 : 0 )
            << "," << reward
            << "," << distance
            << "," << averagePredictedValue
            << "," << steps
            << "," << ( learned ? 1 : 0 )
            << "," << policyEpisodes
            << "," << policyUpdates
            << "," << WinRate()
            << "," << AverageReward()
            << "," << AverageDistance()
            << "," << AveragePredictedValue()
            << "\n";

        WriteSummary( survived, distance, reward, averagePredictedValue, steps, learned, policyEpisodes, policyUpdates );
    }

private:
    TrainingMetrics()
        : episodes_( 0 ),
          wins_( 0 ),
          rewardTotal_( 0 ),
          distanceTotal_( 0 ),
          predictedValueTotal_( 0 ),
          stepsTotal_( 0 ),
          headerChecked_( false ),
          headerNeeded_( true )
    {
    }

    bool NeedHeader()
    {
        if ( headerChecked_ )
        {
            return headerNeeded_;
        }

        headerChecked_ = true;
        std::ifstream in;
        headerNeeded_ = !tDirectories::Var().Open( in, static_cast< char const * >( sg_metricsFile ) );
        return headerNeeded_;
    }

    REAL WinRate() const
    {
        if ( episodes_ <= 0 )
        {
            return 0;
        }

        return static_cast< REAL >( wins_ ) / static_cast< REAL >( episodes_ );
    }

    REAL AverageReward() const
    {
        if ( episodes_ <= 0 )
        {
            return 0;
        }

        return rewardTotal_ / static_cast< REAL >( episodes_ );
    }

    REAL AverageDistance() const
    {
        if ( episodes_ <= 0 )
        {
            return 0;
        }

        return distanceTotal_ / static_cast< REAL >( episodes_ );
    }

    REAL AveragePredictedValue() const
    {
        if ( episodes_ <= 0 )
        {
            return 0;
        }

        return predictedValueTotal_ / static_cast< REAL >( episodes_ );
    }

    REAL AverageSteps() const
    {
        if ( episodes_ <= 0 )
        {
            return 0;
        }

        return static_cast< REAL >( stepsTotal_ ) / static_cast< REAL >( episodes_ );
    }

    void WriteSummary( bool survived, REAL distance, REAL reward, REAL averagePredictedValue, unsigned int steps, bool learned, unsigned int policyEpisodes, unsigned int policyUpdates )
    {
        std::string summaryFile = static_cast< char const * >( sg_metricsFile );
        summaryFile += ".latest";

        std::ofstream out;
        if ( !tDirectories::Var().Open( out, summaryFile.c_str(), std::ios::trunc ) )
        {
            return;
        }

        out.setf( std::ios::fixed );
        out.precision( 6 );
        out << "episodes " << episodes_ << "\n";
        out << "wins " << wins_ << "\n";
        out << "win_rate " << WinRate() << "\n";
        out << "average_reward " << AverageReward() << "\n";
        out << "average_distance " << AverageDistance() << "\n";
        out << "average_predicted_value " << AveragePredictedValue() << "\n";
        out << "average_steps " << AverageSteps() << "\n";
        out << "last_survived " << ( survived ? 1 : 0 ) << "\n";
        out << "last_reward " << reward << "\n";
        out << "last_distance " << distance << "\n";
        out << "last_average_predicted_value " << averagePredictedValue << "\n";
        out << "last_steps " << steps << "\n";
        out << "last_learned " << ( learned ? 1 : 0 ) << "\n";
        out << "policy_episodes " << policyEpisodes << "\n";
        out << "policy_updates " << policyUpdates << "\n";
    }

    unsigned long long episodes_;
    unsigned long long wins_;
    REAL rewardTotal_;
    REAL distanceTotal_;
    REAL predictedValueTotal_;
    unsigned long long stepsTotal_;
    bool headerChecked_;
    bool headerNeeded_;
};

class Policy
{
public:
    enum
    {
        kLiveView = -1
    };

    static Policy & Get()
    {
        // Blacklight is a single shared policy/model instance.
        // Every spawned neural bot reads/writes this same model.
        static Policy policy;
        return policy;
    }

    void EnsureLoaded()
    {
        if ( loaded_ ) return;
        loaded_ = true;
        if ( !Load() )
        {
            Seed();
            dirty_ = true;
            Save( true );
            con << sg_aiName << ": created neural model (" << kParameterCount << " params) in var/" << sg_modelFile << ".\n";
        }
    }

    int AcquireView()
    {
        EnsureLoaded();
        if ( sg_botCount == 1 ) return kLiveView;
        if ( snapshots_.empty() ) return kLiveView;

        REAL historicProb = Clamp( sg_policyHistoricProb, 0.0f, 1.0f );
        if ( RandomUnit() >= historicProb ) return kLiveView;

        int size = static_cast< int >( snapshots_.size() );
        int index = static_cast< int >( RandomUnit() * size );
        if ( index < 0 ) index = 0;
        if ( index >= size ) index = size - 1;
        return index;
    }

    bool IsLiveView( int view ) const
    {
        return view == kLiveView;
    }

    unsigned int Episodes() const
    {
        return episodes_;
    }

    unsigned int Updates() const
    {
        return updates_;
    }

    void SaveNow()
    {
        Save( true );
    }

    int Choose( int viewIndex, REAL const x[kFeatures], bool canLeft, bool canRight, REAL h0[kTemporalHidden], REAL h1[kHidden1], REAL h2[kHidden2], REAL h3[kHidden3], REAL p[kActions], REAL & v )
    {
        EnsureLoaded();
        Forward( GetView( viewIndex ), x, canLeft, canRight, h0, h1, h2, h3, p, v );

        int greedy = 0;
        for ( int a = 1; a < kActions; ++a )
            if ( p[a] > p[greedy] ) greedy = a;

        if ( !sg_learn || !IsLiveView( viewIndex ) || RandomUnit() >= Clamp( sg_exploration, 0.0f, 1.0f ) )
            return greedy;

        REAL r = RandomUnit(), c = 0;
        for ( int a = 0; a < kActions; ++a )
        {
            c += p[a];
            if ( r <= c ) return a;
        }
        return greedy;
    }

    void Train( std::vector< Step > const & episode, REAL reward )
    {
        EnsureLoaded();
        if ( episode.empty() ) return;

        REAL lr = Clamp( sg_learningRate, 0.0f, 1.0f );
        if ( lr <= 0 ) return;

        REAL bDecay = Clamp( sg_baselineDecay, 0.0001f, 1.0f );
        baseline_ = baseline_ * ( 1.0f - bDecay ) + reward * bDecay;

        int steps = static_cast< int >( episode.size() );
        std::vector< REAL > returns( steps, 0 );
        std::vector< REAL > advantages( steps, 0 );
        REAL discount = Clamp( sg_discount, 0.0f, 1.0f );
        REAL discounted = reward;
        for ( int t = steps - 1; t >= 0; --t )
        {
            returns[t] = discounted;
            advantages[t] = returns[t] - episode[t].v;
            discounted *= discount;
        }

        REAL mean = 0;
        for ( int t = 0; t < steps; ++t ) mean += advantages[t];
        mean /= static_cast< REAL >( steps );

        REAL variance = 0;
        for ( int t = 0; t < steps; ++t )
        {
            REAL d = advantages[t] - mean;
            variance += d * d;
        }
        variance /= static_cast< REAL >( steps );
        REAL invStd = variance > 1E-8f ? 1.0f / std::sqrt( variance + 1E-6f ) : 1.0f;

        int epochs = sg_trainEpochs < 1 ? 1 : sg_trainEpochs;
        for ( int epoch = 0; epoch < epochs; ++epoch )
        {
            for ( int t = 0; t < steps; ++t )
            {
                Step const & s = episode[t];
                REAL normalizedAdvantage = ( advantages[t] - mean ) * invStd;
                REAL scale = lr * normalizedAdvantage / static_cast< REAL >( steps * epochs );

                REAL h0[kTemporalHidden];
                REAL h1[kHidden1];
                REAL h2[kHidden2];
                REAL h3[kHidden3];
                REAL p[kActions];
                REAL v = 0;
                Forward( CurrentView(), s.x, s.canLeft, s.canRight, h0, h1, h2, h3, p, v );

                REAL valueError = returns[t] - v;
                REAL valueScale = Clamp( sg_valueLearningRate, 0.0f, 1.0f ) * valueError / static_cast< REAL >( steps * epochs );
                if ( scale == 0 && valueScale == 0 ) continue;

                REAL d4[kActions];
                for ( int a = 0; a < kActions; ++a )
                {
                    REAL target = ( a == s.action ) ? 1.0f : 0.0f;
                    d4[a] = ( target - p[a] ) * scale;
                }

                REAL d3[kHidden3];
                for ( int k = 0; k < kHidden3; ++k )
                {
                    REAL back = valueScale * w5_[k];
                    for ( int a = 0; a < kActions; ++a ) back += d4[a] * w4_[a][k];
                    d3[k] = ( 1.0f - h3[k] * h3[k] ) * back;
                }

                REAL d2[kHidden2];
                for ( int j = 0; j < kHidden2; ++j )
                {
                    REAL back = 0;
                    for ( int k = 0; k < kHidden3; ++k ) back += d3[k] * w3_[k][j];
                    d2[j] = ( 1.0f - h2[j] * h2[j] ) * back;
                }

                REAL d1[kHidden1];
                for ( int j = 0; j < kHidden1; ++j )
                {
                    REAL back = 0;
                    for ( int k = 0; k < kHidden2; ++k ) back += d2[k] * w2_[k][j];
                    d1[j] = ( 1.0f - h1[j] * h1[j] ) * back;
                }

                REAL d0[kTemporalHidden];
                for ( int j = 0; j < kTemporalHidden; ++j )
                {
                    REAL back = 0;
                    for ( int k = 0; k < kHidden1; ++k ) back += d1[k] * w1_[k][j];
                    d0[j] = ( 1.0f - h0[j] * h0[j] ) * back;
                }

                for ( int a = 0; a < kActions; ++a )
                {
                    b4_[a] += d4[a];
                    for ( int k = 0; k < kHidden3; ++k ) w4_[a][k] += d4[a] * h3[k];
                }

                b5_ += valueScale;
                for ( int k = 0; k < kHidden3; ++k ) w5_[k] += valueScale * h3[k];

                for ( int k = 0; k < kHidden3; ++k )
                {
                    b3_[k] += d3[k];
                    for ( int j = 0; j < kHidden2; ++j ) w3_[k][j] += d3[k] * h2[j];
                }

                for ( int k = 0; k < kHidden2; ++k )
                {
                    b2_[k] += d2[k];
                    for ( int j = 0; j < kHidden1; ++j ) w2_[k][j] += d2[k] * h1[j];
                }

                for ( int j = 0; j < kHidden1; ++j )
                {
                    b1_[j] += d1[j];
                    for ( int i = 0; i < kTemporalHidden; ++i ) w1_[j][i] += d1[j] * h0[i];
                }

                for ( int j = 0; j < kTemporalHidden; ++j )
                {
                    b0_[j] += d0[j];
                    for ( int i = 0; i < kFeatures; ++i ) w0_[j][i] += d0[j] * s.x[i];
                }
            }
        }

        REAL shrink = 1.0f - lr * Clamp( sg_weightDecay, 0.0f, 1.0f );
        for ( int j = 0; j < kTemporalHidden; ++j )
            for ( int i = 0; i < kFeatures; ++i )
                w0_[j][i] = Clamp( w0_[j][i] * shrink, -sg_weightClip, sg_weightClip );

        for ( int j = 0; j < kHidden1; ++j )
            for ( int i = 0; i < kTemporalHidden; ++i )
                w1_[j][i] = Clamp( w1_[j][i] * shrink, -sg_weightClip, sg_weightClip );

        for ( int k = 0; k < kHidden2; ++k )
            for ( int j = 0; j < kHidden1; ++j )
                w2_[k][j] = Clamp( w2_[k][j] * shrink, -sg_weightClip, sg_weightClip );

        for ( int k = 0; k < kHidden3; ++k )
            for ( int j = 0; j < kHidden2; ++j )
                w3_[k][j] = Clamp( w3_[k][j] * shrink, -sg_weightClip, sg_weightClip );

        for ( int a = 0; a < kActions; ++a )
            for ( int k = 0; k < kHidden3; ++k )
                w4_[a][k] = Clamp( w4_[a][k] * shrink, -sg_weightClip, sg_weightClip );

        for ( int k = 0; k < kHidden3; ++k )
            w5_[k] = Clamp( w5_[k] * shrink, -sg_weightClip, sg_weightClip );

        ++episodes_;
        ++updates_;
        dirty_ = true;
        MaybeAddSnapshot();
        MaybeWriteCheckpoint();
        if ( sg_saveEvery < 1 ) sg_saveEvery = 1;
        if ( episodes_ % static_cast< unsigned int >( sg_saveEvery ) == 0 ) Save();
    }

private:
    struct WeightsView
    {
        REAL const ( *w0 )[kFeatures];
        REAL const * b0;
        REAL const ( *w1 )[kTemporalHidden];
        REAL const * b1;
        REAL const ( *w2 )[kHidden1];
        REAL const * b2;
        REAL const ( *w3 )[kHidden2];
        REAL const * b3;
        REAL const ( *w4 )[kHidden3];
        REAL const * b4;
        REAL const * w5;
        REAL b5;
    };

    struct Snapshot
    {
        REAL w0[kTemporalHidden][kFeatures];
        REAL b0[kTemporalHidden];
        REAL w1[kHidden1][kTemporalHidden];
        REAL b1[kHidden1];
        REAL w2[kHidden2][kHidden1];
        REAL b2[kHidden2];
        REAL w3[kHidden3][kHidden2];
        REAL b3[kHidden3];
        REAL w4[kActions][kHidden3];
        REAL b4[kActions];
        REAL w5[kHidden3];
        REAL b5;
        unsigned int episode;
    };

    Policy(): baseline_(0), episodes_(0), updates_(0), loaded_(false), dirty_(false)
    {
        for ( int j = 0; j < kTemporalHidden; ++j )
        {
            b0_[j] = 0;
            for ( int i = 0; i < kFeatures; ++i ) w0_[j][i] = 0;
        }
        for ( int j = 0; j < kHidden1; ++j )
        {
            b1_[j] = 0;
            for ( int i = 0; i < kTemporalHidden; ++i ) w1_[j][i] = 0;
        }
        for ( int k = 0; k < kHidden2; ++k )
        {
            b2_[k] = 0;
            for ( int j = 0; j < kHidden1; ++j ) w2_[k][j] = 0;
        }
        for ( int k = 0; k < kHidden3; ++k )
        {
            b3_[k] = 0;
            for ( int j = 0; j < kHidden2; ++j ) w3_[k][j] = 0;
        }
        for ( int a = 0; a < kActions; ++a )
        {
            b4_[a] = 0;
            for ( int k = 0; k < kHidden3; ++k ) w4_[a][k] = 0;
        }
        b5_ = 0;
        for ( int k = 0; k < kHidden3; ++k ) w5_[k] = 0;
    }

    WeightsView CurrentView() const
    {
        WeightsView view;
        view.w0 = w0_;
        view.b0 = b0_;
        view.w1 = w1_;
        view.b1 = b1_;
        view.w2 = w2_;
        view.b2 = b2_;
        view.w3 = w3_;
        view.b3 = b3_;
        view.w4 = w4_;
        view.b4 = b4_;
        view.w5 = w5_;
        view.b5 = b5_;
        return view;
    }

    WeightsView GetView( int viewIndex ) const
    {
        if ( viewIndex >= 0 && viewIndex < static_cast< int >( snapshots_.size() ) )
        {
            Snapshot const & snapshot = snapshots_[viewIndex];
            WeightsView view;
            view.w0 = snapshot.w0;
            view.b0 = snapshot.b0;
            view.w1 = snapshot.w1;
            view.b1 = snapshot.b1;
            view.w2 = snapshot.w2;
            view.b2 = snapshot.b2;
            view.w3 = snapshot.w3;
            view.b3 = snapshot.b3;
            view.w4 = snapshot.w4;
            view.b4 = snapshot.b4;
            view.w5 = snapshot.w5;
            view.b5 = snapshot.b5;
            return view;
        }

        return CurrentView();
    }

    void MaybeAddSnapshot()
    {
        if ( sg_policyPoolSize <= 0 ) return;
        if ( sg_policySnapshotEvery <= 0 ) return;
        if ( episodes_ < static_cast< unsigned int >( sg_policySnapshotWarmup < 0 ? 0 : sg_policySnapshotWarmup ) ) return;
        if ( episodes_ % static_cast< unsigned int >( sg_policySnapshotEvery ) != 0 ) return;

        if ( static_cast< int >( snapshots_.size() ) >= sg_policyPoolSize )
        {
            snapshots_.erase( snapshots_.begin() );
        }
        snapshots_.push_back( Snapshot() );
        Snapshot & snapshot = snapshots_.back();
        snapshot.episode = episodes_;
        for ( int j = 0; j < kTemporalHidden; ++j )
        {
            snapshot.b0[j] = b0_[j];
            for ( int i = 0; i < kFeatures; ++i ) snapshot.w0[j][i] = w0_[j][i];
        }
        for ( int j = 0; j < kHidden1; ++j )
        {
            snapshot.b1[j] = b1_[j];
            for ( int i = 0; i < kTemporalHidden; ++i ) snapshot.w1[j][i] = w1_[j][i];
        }
        for ( int k = 0; k < kHidden2; ++k )
        {
            snapshot.b2[k] = b2_[k];
            for ( int j = 0; j < kHidden1; ++j ) snapshot.w2[k][j] = w2_[k][j];
        }
        for ( int k = 0; k < kHidden3; ++k )
        {
            snapshot.b3[k] = b3_[k];
            for ( int j = 0; j < kHidden2; ++j ) snapshot.w3[k][j] = w3_[k][j];
        }
        for ( int a = 0; a < kActions; ++a )
        {
            snapshot.b4[a] = b4_[a];
            for ( int k = 0; k < kHidden3; ++k ) snapshot.w4[a][k] = w4_[a][k];
        }
        snapshot.b5 = b5_;
        for ( int k = 0; k < kHidden3; ++k ) snapshot.w5[k] = w5_[k];
    }

    void MaybeWriteCheckpoint() const
    {
        char const * checkpointPrefix = static_cast< char const * >( sg_checkpointPrefix );
        if ( !checkpointPrefix || !checkpointPrefix[0] )
        {
            return;
        }
        if ( sg_checkpointEvery < 1 )
        {
            return;
        }
        if ( episodes_ % static_cast< unsigned int >( sg_checkpointEvery ) != 0 )
        {
            return;
        }

        std::ostringstream checkpointName;
        checkpointName << checkpointPrefix << "_ep" << std::setw( 8 ) << std::setfill( '0' ) << episodes_ << ".txt";
        SaveToFile( checkpointName.str().c_str() );

        std::ostringstream latestName;
        latestName << checkpointPrefix << "_latest.txt";
        std::ofstream latest;
        if ( tDirectories::Var().Open( latest, latestName.str().c_str(), std::ios::trunc ) )
        {
            latest << checkpointName.str() << "\n";
            latest << episodes_ << "\n";
        }
    }

    void Forward( WeightsView const & view, REAL const x[kFeatures], bool canLeft, bool canRight, REAL h0[kTemporalHidden], REAL h1[kHidden1], REAL h2[kHidden2], REAL h3[kHidden3], REAL p[kActions], REAL & v ) const
    {
        for ( int j = 0; j < kTemporalHidden; ++j )
        {
            REAL sum = view.b0[j];
            for ( int i = 0; i < kFeatures; ++i ) sum += view.w0[j][i] * x[i];
            h0[j] = std::tanh( sum );
        }

        for ( int j = 0; j < kHidden1; ++j )
        {
            REAL sum = view.b1[j];
            for ( int i = 0; i < kTemporalHidden; ++i ) sum += view.w1[j][i] * h0[i];
            h1[j] = std::tanh( sum );
        }

        for ( int k = 0; k < kHidden2; ++k )
        {
            REAL sum = view.b2[k];
            for ( int j = 0; j < kHidden1; ++j ) sum += view.w2[k][j] * h1[j];
            h2[k] = std::tanh( sum );
        }

        for ( int k = 0; k < kHidden3; ++k )
        {
            REAL sum = view.b3[k];
            for ( int j = 0; j < kHidden2; ++j ) sum += view.w3[k][j] * h2[j];
            h3[k] = std::tanh( sum );
        }

        REAL logits[kActions];
        for ( int a = 0; a < kActions; ++a )
        {
            REAL sum = view.b4[a];
            for ( int k = 0; k < kHidden3; ++k ) sum += view.w4[a][k] * h3[k];
            logits[a] = sum;
        }
        if ( !canLeft ) logits[0] = -1E+20f;
        if ( !canRight ) logits[2] = -1E+20f;

        REAL maxLogit = logits[0];
        for ( int a = 1; a < kActions; ++a ) if ( logits[a] > maxLogit ) maxLogit = logits[a];

        REAL sumExp = 0;
        for ( int a = 0; a < kActions; ++a )
        {
            p[a] = logits[a] < -1E+10f ? 0.0f : std::exp( logits[a] - maxLogit );
            sumExp += p[a];
        }
        if ( sumExp <= 0 )
        {
            p[0] = canLeft ? 0.5f : 0.0f;
            p[1] = 1.0f;
            p[2] = canRight ? 0.5f : 0.0f;
            sumExp = p[0] + p[1] + p[2];
        }
        for ( int a = 0; a < kActions; ++a ) p[a] /= sumExp;

        REAL value = view.b5;
        for ( int k = 0; k < kHidden3; ++k ) value += view.w5[k] * h3[k];
        v = value;
    }

    bool Load()
    {
        std::ifstream in;
        if ( !tDirectories::Var().Open( in, static_cast< char const * >( sg_modelFile ) ) ) return false;

        std::string magic;
        int f = 0, h0 = 0, h1 = 0, h2 = 0, h3 = 0, a = 0;
        in >> magic >> f >> h0 >> h1 >> h2 >> h3 >> a;
        if ( magic != "ARMAGETRON_TRAINED_AI_NN_V10" ||
             f != kFeatures || h0 != kTemporalHidden || h1 != kHidden1 || h2 != kHidden2 || h3 != kHidden3 || a != kActions ) return false;
        in >> baseline_ >> episodes_ >> updates_;
        if ( in.fail() ) return false;

        for ( int j = 0; j < kTemporalHidden; ++j )
        {
            for ( int i = 0; i < kFeatures; ++i ) in >> w0_[j][i];
            in >> b0_[j];
        }
        for ( int j = 0; j < kHidden1; ++j )
        {
            for ( int i = 0; i < kTemporalHidden; ++i ) in >> w1_[j][i];
            in >> b1_[j];
        }
        for ( int k = 0; k < kHidden2; ++k )
        {
            for ( int j = 0; j < kHidden1; ++j ) in >> w2_[k][j];
            in >> b2_[k];
        }
        for ( int k = 0; k < kHidden3; ++k )
        {
            for ( int j = 0; j < kHidden2; ++j ) in >> w3_[k][j];
            in >> b3_[k];
        }
        for ( int ac = 0; ac < kActions; ++ac )
        {
            for ( int k = 0; k < kHidden3; ++k ) in >> w4_[ac][k];
            in >> b4_[ac];
        }
        for ( int k = 0; k < kHidden3; ++k ) in >> w5_[k];
        in >> b5_;
        return !in.fail();
    }

    bool SaveToFile( char const * fileName ) const
    {
        std::ofstream out;
        if ( !tDirectories::Var().Open( out, fileName, std::ios::trunc ) ) return false;

        out.setf( std::ios::fixed );
        out.precision( 9 );
        out << "ARMAGETRON_TRAINED_AI_NN_V10\n";
        out << kFeatures << " " << kTemporalHidden << " " << kHidden1 << " " << kHidden2 << " " << kHidden3 << " " << kActions << "\n";
        out << baseline_ << " " << episodes_ << " " << updates_ << "\n";
        for ( int j = 0; j < kTemporalHidden; ++j )
        {
            for ( int i = 0; i < kFeatures; ++i ) out << w0_[j][i] << " ";
            out << b0_[j] << "\n";
        }
        for ( int j = 0; j < kHidden1; ++j )
        {
            for ( int i = 0; i < kTemporalHidden; ++i ) out << w1_[j][i] << " ";
            out << b1_[j] << "\n";
        }
        for ( int k = 0; k < kHidden2; ++k )
        {
            for ( int j = 0; j < kHidden1; ++j ) out << w2_[k][j] << " ";
            out << b2_[k] << "\n";
        }
        for ( int k = 0; k < kHidden3; ++k )
        {
            for ( int j = 0; j < kHidden2; ++j ) out << w3_[k][j] << " ";
            out << b3_[k] << "\n";
        }
        for ( int ac = 0; ac < kActions; ++ac )
        {
            for ( int k = 0; k < kHidden3; ++k ) out << w4_[ac][k] << " ";
            out << b4_[ac] << "\n";
        }
        for ( int k = 0; k < kHidden3; ++k ) out << w5_[k] << " ";
        out << b5_ << "\n";
        return !out.fail();
    }

    void Save( bool force = false )
    {
        if ( !dirty_ && !force ) return;
        if ( SaveToFile( static_cast< char const * >( sg_modelFile ) ) )
        {
            dirty_ = false;
        }
    }

    void Seed()
    {
        for ( int j = 0; j < kTemporalHidden; ++j )
        {
            b0_[j] = RandomSigned() * 0.05f;
            for ( int i = 0; i < kFeatures; ++i ) w0_[j][i] = RandomSigned() * 0.05f;
        }
        for ( int j = 0; j < kHidden1; ++j )
        {
            b1_[j] = RandomSigned() * 0.05f;
            for ( int i = 0; i < kTemporalHidden; ++i ) w1_[j][i] = RandomSigned() * 0.05f;
        }
        for ( int k = 0; k < kHidden2; ++k )
        {
            b2_[k] = RandomSigned() * 0.05f;
            for ( int j = 0; j < kHidden1; ++j ) w2_[k][j] = RandomSigned() * 0.05f;
        }
        for ( int k = 0; k < kHidden3; ++k )
        {
            b3_[k] = RandomSigned() * 0.05f;
            for ( int j = 0; j < kHidden2; ++j ) w3_[k][j] = RandomSigned() * 0.05f;
        }
        for ( int ac = 0; ac < kActions; ++ac )
        {
            b4_[ac] = 0;
            for ( int k = 0; k < kHidden3; ++k ) w4_[ac][k] = RandomSigned() * 0.05f;
        }
        b5_ = 0;
        for ( int k = 0; k < kHidden3; ++k ) w5_[k] = RandomSigned() * 0.05f;

        for ( int i = 0; i < kBaseFeatures && i < kTemporalHidden; ++i )
        {
            for ( int k = 0; k < kFeatures; ++k ) w0_[i][k] = 0;
            w0_[i][kCurrentFrameOffset + i] = 1.5f;
            b0_[i] = 0;
        }

        for ( int i = 0; i < kBaseFeatures && i < kHidden1; ++i )
        {
            for ( int j = 0; j < kTemporalHidden; ++j ) w1_[i][j] = 0;
            w1_[i][i] = 1.0f;
            b1_[i] = 0;
        }

        for ( int i = 0; i < kBaseFeatures && i < kHidden2; ++i )
        {
            for ( int j = 0; j < kHidden1; ++j ) w2_[i][j] = 0;
            w2_[i][i] = 1.0f;
            b2_[i] = 0;
        }

        for ( int i = 0; i < kBaseFeatures && i < kHidden3; ++i )
        {
            for ( int j = 0; j < kHidden2; ++j ) w3_[i][j] = 0;
            w3_[i][i] = 1.0f;
            b3_[i] = 0;
        }

        w4_[1][kFront] += 1.8f;
        w4_[1][kFrontNarrowLeft] += 1.0f;
        w4_[1][kFrontNarrowRight] += 1.0f;
        w4_[1][kFrontLeft] += 0.6f;
        w4_[1][kFrontRight] += 0.6f;
        w4_[1][kFrontPressure] += -1.2f;
        w4_[1][kFrontRimWall] += -1.0f;

        w4_[0][kLeft] += 1.8f;
        w4_[0][kFrontLeft] += 1.3f;
        w4_[0][kWideLeft] += 1.0f;
        w4_[0][kBackLeft] += 0.8f;
        w4_[0][kEscapeLeft] += 1.2f;
        w4_[0][kFront] += -1.2f;
        w4_[0][kCanLeft] += 1.0f;
        w4_[0][kLeftRimWall] += -0.8f;

        w4_[2][kRight] += 1.8f;
        w4_[2][kFrontRight] += 1.3f;
        w4_[2][kWideRight] += 1.0f;
        w4_[2][kBackRight] += 0.8f;
        w4_[2][kEscapeRight] += 1.2f;
        w4_[2][kFront] += -1.2f;
        w4_[2][kCanRight] += 1.0f;
        w4_[2][kRightRimWall] += -0.8f;

        w5_[kFront] += 1.0f;
        w5_[kForwardArcSafety] += 0.8f;
        w5_[kWallCrowding] += -1.0f;
        w5_[kEnemyNear] += -0.6f;
        w5_[kEnemyFrontProximity] += -0.5f;
        w5_[kEscapeLeft] += 0.4f;
        w5_[kEscapeRight] += 0.4f;
    }

    REAL w0_[kTemporalHidden][kFeatures];
    REAL b0_[kTemporalHidden];
    REAL w1_[kHidden1][kTemporalHidden];
    REAL b1_[kHidden1];
    REAL w2_[kHidden2][kHidden1];
    REAL b2_[kHidden2];
    REAL w3_[kHidden3][kHidden2];
    REAL b3_[kHidden3];
    REAL w4_[kActions][kHidden3];
    REAL b4_[kActions];
    REAL w5_[kHidden3];
    REAL b5_;
    REAL baseline_;
    unsigned int episodes_;
    unsigned int updates_;
    bool loaded_;
    bool dirty_;
    std::vector< Snapshot > snapshots_;
};

struct OfflineEpisodeData
{
    OfflineEpisodeData()
        : survived( false ),
          distance( 0 ),
          reward( 0 )
    {
    }

    bool survived;
    REAL distance;
    REAL reward;
    std::vector< Step > steps;
};

class OfflineTrainerState
{
public:
    OfflineTrainerState()
        : metricsEpisodes_( 0 ),
          wins_( 0 ),
          rewardTotal_( 0 ),
          distanceTotal_( 0 ),
          predictedValueTotal_( 0 ),
          stepsTotal_( 0 ),
          lastSurvived_( false ),
          lastReward_( 0 ),
          lastDistance_( 0 ),
          lastAveragePredictedValue_( 0 ),
          lastSteps_( 0 ),
          lastLearned_( false )
    {
    }

    void EnsureSource( std::string const & path )
    {
        if ( offsets_.find( path ) == offsets_.end() )
        {
            offsets_[path] = 0;
        }
    }

    long long OffsetFor( std::string const & path ) const
    {
        std::map< std::string, long long >::const_iterator it = offsets_.find( path );
        if ( it == offsets_.end() )
        {
            return 0;
        }
        return it->second;
    }

    void SetOffset( std::string const & path, long long offset )
    {
        offsets_[path] = offset < 0 ? 0 : offset;
    }

    void RecordEpisode( bool survived, REAL distance, REAL reward, REAL averagePredictedValue, unsigned int steps, bool learned )
    {
        ++metricsEpisodes_;
        if ( survived )
        {
            ++wins_;
        }
        rewardTotal_ += reward;
        distanceTotal_ += distance;
        predictedValueTotal_ += averagePredictedValue;
        stepsTotal_ += steps;
        lastSurvived_ = survived;
        lastReward_ = reward;
        lastDistance_ = distance;
        lastAveragePredictedValue_ = averagePredictedValue;
        lastSteps_ = steps;
        lastLearned_ = learned;
    }

    unsigned long long MetricsEpisodes() const { return metricsEpisodes_; }
    unsigned long long Wins() const { return wins_; }
    REAL RewardTotal() const { return rewardTotal_; }
    REAL DistanceTotal() const { return distanceTotal_; }
    REAL PredictedValueTotal() const { return predictedValueTotal_; }
    unsigned long long StepsTotal() const { return stepsTotal_; }
    bool LastSurvived() const { return lastSurvived_; }
    REAL LastReward() const { return lastReward_; }
    REAL LastDistance() const { return lastDistance_; }
    REAL LastAveragePredictedValue() const { return lastAveragePredictedValue_; }
    unsigned int LastSteps() const { return lastSteps_; }
    bool LastLearned() const { return lastLearned_; }

    REAL WinRate() const
    {
        if ( metricsEpisodes_ <= 0 )
        {
            return 0;
        }
        return static_cast< REAL >( wins_ ) / static_cast< REAL >( metricsEpisodes_ );
    }

    REAL AverageReward() const
    {
        if ( metricsEpisodes_ <= 0 )
        {
            return 0;
        }
        return rewardTotal_ / static_cast< REAL >( metricsEpisodes_ );
    }

    REAL AverageDistance() const
    {
        if ( metricsEpisodes_ <= 0 )
        {
            return 0;
        }
        return distanceTotal_ / static_cast< REAL >( metricsEpisodes_ );
    }

    REAL AveragePredictedValue() const
    {
        if ( metricsEpisodes_ <= 0 )
        {
            return 0;
        }
        return predictedValueTotal_ / static_cast< REAL >( metricsEpisodes_ );
    }

    REAL AverageSteps() const
    {
        if ( metricsEpisodes_ <= 0 )
        {
            return 0;
        }
        return static_cast< REAL >( stepsTotal_ ) / static_cast< REAL >( metricsEpisodes_ );
    }

    bool Load()
    {
        char const * stateFile = static_cast< char const * >( sg_offlineStateFile );
        if ( !stateFile || !stateFile[0] )
        {
            return false;
        }

        std::ifstream in;
        if ( !tDirectories::Var().Open( in, stateFile ) )
        {
            return false;
        }

        std::string magic;
        in >> magic;
        if ( magic != "BLACKLIGHT_OFFLINE_TRAIN_STATE_V1" )
        {
            return false;
        }

        std::string key;
        while ( in >> key )
        {
            if ( key == "metrics_episodes" )
            {
                in >> metricsEpisodes_;
            }
            else if ( key == "wins" )
            {
                in >> wins_;
            }
            else if ( key == "reward_total" )
            {
                in >> rewardTotal_;
            }
            else if ( key == "distance_total" )
            {
                in >> distanceTotal_;
            }
            else if ( key == "predicted_value_total" )
            {
                in >> predictedValueTotal_;
            }
            else if ( key == "steps_total" )
            {
                in >> stepsTotal_;
            }
            else if ( key == "last_survived" )
            {
                int value = 0;
                in >> value;
                lastSurvived_ = value != 0;
            }
            else if ( key == "last_reward" )
            {
                in >> lastReward_;
            }
            else if ( key == "last_distance" )
            {
                in >> lastDistance_;
            }
            else if ( key == "last_average_predicted_value" )
            {
                in >> lastAveragePredictedValue_;
            }
            else if ( key == "last_steps" )
            {
                in >> lastSteps_;
            }
            else if ( key == "last_learned" )
            {
                int value = 0;
                in >> value;
                lastLearned_ = value != 0;
            }
            else if ( key == "source" )
            {
                std::string path;
                long long offset = 0;
                in >> path >> offset;
                offsets_[path] = offset;
            }
        }

        return !in.fail();
    }

    void Save() const
    {
        char const * stateFile = static_cast< char const * >( sg_offlineStateFile );
        if ( !stateFile || !stateFile[0] )
        {
            return;
        }

        std::ofstream out;
        if ( !tDirectories::Var().Open( out, stateFile, std::ios::trunc ) )
        {
            return;
        }

        out.setf( std::ios::fixed );
        out.precision( 9 );
        out << "BLACKLIGHT_OFFLINE_TRAIN_STATE_V1\n";
        out << "metrics_episodes " << metricsEpisodes_ << "\n";
        out << "wins " << wins_ << "\n";
        out << "reward_total " << rewardTotal_ << "\n";
        out << "distance_total " << distanceTotal_ << "\n";
        out << "predicted_value_total " << predictedValueTotal_ << "\n";
        out << "steps_total " << stepsTotal_ << "\n";
        out << "last_survived " << ( lastSurvived_ ? 1 : 0 ) << "\n";
        out << "last_reward " << lastReward_ << "\n";
        out << "last_distance " << lastDistance_ << "\n";
        out << "last_average_predicted_value " << lastAveragePredictedValue_ << "\n";
        out << "last_steps " << lastSteps_ << "\n";
        out << "last_learned " << ( lastLearned_ ? 1 : 0 ) << "\n";

        for ( std::map< std::string, long long >::const_iterator it = offsets_.begin(); it != offsets_.end(); ++it )
        {
            out << "source " << it->first << " " << it->second << "\n";
        }
    }

private:
    unsigned long long metricsEpisodes_;
    unsigned long long wins_;
    REAL rewardTotal_;
    REAL distanceTotal_;
    REAL predictedValueTotal_;
    unsigned long long stepsTotal_;
    bool lastSurvived_;
    REAL lastReward_;
    REAL lastDistance_;
    REAL lastAveragePredictedValue_;
    unsigned int lastSteps_;
    bool lastLearned_;
    std::map< std::string, long long > offsets_;
};

static bool OfflineMetricsNeedsHeader()
{
    char const * metricsFile = static_cast< char const * >( sg_metricsFile );
    if ( !metricsFile || !metricsFile[0] )
    {
        return false;
    }

    std::ifstream in;
    if ( !tDirectories::Var().Open( in, metricsFile ) )
    {
        return true;
    }

    return in.peek() == std::ifstream::traits_type::eof();
}

static void AppendOfflineMetricsCsv( OfflineTrainerState const & state, bool survived, REAL distance, REAL reward, REAL averagePredictedValue, unsigned int steps, bool learned, unsigned int policyEpisodes, unsigned int policyUpdates )
{
    char const * metricsFile = static_cast< char const * >( sg_metricsFile );
    if ( !metricsFile || !metricsFile[0] )
    {
        return;
    }

    bool writeHeader = OfflineMetricsNeedsHeader();

    std::ofstream out;
    if ( !tDirectories::Var().Open( out, metricsFile, std::ios::app ) )
    {
        return;
    }

    out.setf( std::ios::fixed );
    out.precision( 6 );

    if ( writeHeader )
    {
        out << "episode,survived,reward,distance,average_predicted_value,steps,learned,policy_episodes,policy_updates,cumulative_win_rate,cumulative_average_reward,cumulative_average_distance,cumulative_average_predicted_value\n";
    }

    out << state.MetricsEpisodes()
        << "," << ( survived ? 1 : 0 )
        << "," << reward
        << "," << distance
        << "," << averagePredictedValue
        << "," << steps
        << "," << ( learned ? 1 : 0 )
        << "," << policyEpisodes
        << "," << policyUpdates
        << "," << state.WinRate()
        << "," << state.AverageReward()
        << "," << state.AverageDistance()
        << "," << state.AveragePredictedValue()
        << "\n";
}

static void WriteOfflineMetricsSummary( OfflineTrainerState const & state, unsigned int policyEpisodes, unsigned int policyUpdates )
{
    char const * metricsFile = static_cast< char const * >( sg_metricsFile );
    if ( !metricsFile || !metricsFile[0] )
    {
        return;
    }

    std::string summaryFile = metricsFile;
    summaryFile += ".latest";

    std::ofstream out;
    if ( !tDirectories::Var().Open( out, summaryFile.c_str(), std::ios::trunc ) )
    {
        return;
    }

    out.setf( std::ios::fixed );
    out.precision( 6 );
    out << "episodes " << state.MetricsEpisodes() << "\n";
    out << "wins " << state.Wins() << "\n";
    out << "win_rate " << state.WinRate() << "\n";
    out << "average_reward " << state.AverageReward() << "\n";
    out << "average_distance " << state.AverageDistance() << "\n";
    out << "average_predicted_value " << state.AveragePredictedValue() << "\n";
    out << "average_steps " << state.AverageSteps() << "\n";
    out << "last_survived " << ( state.LastSurvived() ? 1 : 0 ) << "\n";
    out << "last_reward " << state.LastReward() << "\n";
    out << "last_distance " << state.LastDistance() << "\n";
    out << "last_average_predicted_value " << state.LastAveragePredictedValue() << "\n";
    out << "last_steps " << state.LastSteps() << "\n";
    out << "last_learned " << ( state.LastLearned() ? 1 : 0 ) << "\n";
    out << "policy_episodes " << policyEpisodes << "\n";
    out << "policy_updates " << policyUpdates << "\n";
}

static bool LoadOfflineSourceList( std::vector< std::string > & sources )
{
    sources.clear();

    char const * sourceList = static_cast< char const * >( sg_offlineSourceList );
    if ( !sourceList || !sourceList[0] )
    {
        return false;
    }

    std::ifstream in;
    if ( !tDirectories::Var().Open( in, sourceList ) )
    {
        return false;
    }

    std::string line;
    while ( std::getline( in, line ) )
    {
        if ( line.empty() ) continue;
        if ( line[0] == '#' ) continue;
        sources.push_back( line );
    }

    return !sources.empty();
}

static bool ParseOfflineRecordedStep( std::string const & line, unsigned long long & episodeId, bool & survived, REAL & distance, REAL & reward, Step & step )
{
    std::istringstream in( line );
    unsigned int stepIndex = 0;
    int survivedInt = 0;

    if ( !( in >> episodeId >> stepIndex >> survivedInt >> distance >> reward >> step.action ) )
    {
        return false;
    }

    std::vector< REAL > tail;
    REAL value = 0;
    while ( in >> value )
    {
        tail.push_back( value );
    }

    std::size_t oldExpected = static_cast< std::size_t >( kActions + 1 + kFeatures );
    std::size_t newExpected = static_cast< std::size_t >( 2 + kActions + 1 + kFeatures );
    if ( tail.size() != oldExpected && tail.size() != newExpected )
    {
        return false;
    }

    std::size_t index = 0;
    if ( tail.size() == newExpected )
    {
        step.canLeft = tail[index++] > 0.5f;
        step.canRight = tail[index++] > 0.5f;
    }
    else
    {
        step.canLeft = true;
        step.canRight = true;
    }

    for ( int a = 0; a < kActions; ++a )
    {
        step.p[a] = tail[index++];
    }

    step.v = tail[index++];
    for ( int f = 0; f < kFeatures; ++f )
    {
        step.x[f] = tail[index++];
    }

    survived = survivedInt != 0;
    return true;
}

static long long VarFileSize( std::string const & path )
{
    std::ifstream in;
    if ( !tDirectories::Var().Open( in, path.c_str() ) )
    {
        return 0;
    }

    in.seekg( 0, std::ios::end );
    std::ifstream::pos_type end = in.tellg();
    if ( end < 0 )
    {
        return 0;
    }

    return static_cast< long long >( end );
}

static void TrainOfflineEpisode( OfflineEpisodeData const & episodeData, OfflineTrainerState & state )
{
    if ( episodeData.steps.empty() )
    {
        return;
    }

    REAL averagePredictedValue = AveragePredictedValue( episodeData.steps );
    Policy::Get().Train( episodeData.steps, episodeData.reward );
    state.RecordEpisode(
        episodeData.survived,
        episodeData.distance,
        episodeData.reward,
        averagePredictedValue,
        static_cast< unsigned int >( episodeData.steps.size() ),
        true );
    AppendOfflineMetricsCsv(
        state,
        episodeData.survived,
        episodeData.distance,
        episodeData.reward,
        averagePredictedValue,
        static_cast< unsigned int >( episodeData.steps.size() ),
        true,
        Policy::Get().Episodes(),
        Policy::Get().Updates() );
}

static unsigned int TrainOfflineSource( std::string const & path, OfflineTrainerState & state )
{
    std::ifstream in;
    if ( !tDirectories::Var().Open( in, path.c_str() ) )
    {
        return 0;
    }

    long long size = VarFileSize( path );
    long long offset = state.OffsetFor( path );
    if ( offset < 0 || offset > size )
    {
        offset = 0;
    }

    in.seekg( offset, std::ios::beg );

    OfflineEpisodeData currentEpisode;
    unsigned long long currentEpisodeId = 0;
    bool hasEpisode = false;
    unsigned int trainedEpisodes = 0;
    std::string line;

    while ( std::getline( in, line ) )
    {
        if ( line.empty() )
        {
            continue;
        }

        Step step;
        bool survived = false;
        REAL distance = 0;
        REAL reward = 0;
        unsigned long long episodeId = 0;
        if ( !ParseOfflineRecordedStep( line, episodeId, survived, distance, reward, step ) )
        {
            continue;
        }

        if ( !hasEpisode )
        {
            hasEpisode = true;
            currentEpisodeId = episodeId;
            currentEpisode.survived = survived;
            currentEpisode.distance = distance;
            currentEpisode.reward = reward;
            currentEpisode.steps.clear();
        }
        else if ( episodeId != currentEpisodeId )
        {
            TrainOfflineEpisode( currentEpisode, state );
            ++trainedEpisodes;
            currentEpisodeId = episodeId;
            currentEpisode.survived = survived;
            currentEpisode.distance = distance;
            currentEpisode.reward = reward;
            currentEpisode.steps.clear();
        }

        currentEpisode.steps.push_back( step );
    }

    if ( hasEpisode && !currentEpisode.steps.empty() )
    {
        TrainOfflineEpisode( currentEpisode, state );
        ++trainedEpisodes;
    }

    state.SetOffset( path, size );
    return trainedEpisodes;
}

static void RunOfflineTrainerIfConfigured()
{
    if ( !sg_offlineTrain )
    {
        return;
    }

    Policy::Get().EnsureLoaded();

    OfflineTrainerState state;
    state.Load();

    std::vector< std::string > sources;
    LoadOfflineSourceList( sources );

    for ( std::vector< std::string >::const_iterator it = sources.begin(); it != sources.end(); ++it )
    {
        state.EnsureSource( *it );
    }

    unsigned int processedEpisodes = 0;
    for ( std::vector< std::string >::const_iterator it = sources.begin(); it != sources.end(); ++it )
    {
        processedEpisodes += TrainOfflineSource( *it, state );
    }

    Policy::Get().SaveNow();
    state.Save();
    WriteOfflineMetricsSummary( state, Policy::Get().Episodes(), Policy::Get().Updates() );

    con << sg_aiName << ": offline trainer processed " << processedEpisodes
        << " episodes from " << sources.size()
        << " sources. Policy episodes=" << Policy::Get().Episodes()
        << " updates=" << Policy::Get().Updates() << ".\n";

    std::exit( 0 );
}

static eCoord NormalizedDir( eCoord direction )
{
    REAL normSq = direction.NormSquared();
    return normSq <= .000001f ? eCoord( 0, 1 ) : direction * ( 1.0f / std::sqrt( normSq ) );
}

static REAL SenseDistance( gCycle * cycle, eCoord const & direction, REAL lookAhead, gSensorWallType * wallType = 0 )
{
    gSensor sensor( cycle, cycle->Position(), NormalizedDir( direction ) );
    sensor.detect( lookAhead );
    if ( wallType ) *wallType = sensor.type;
    return sensor.ehit ? sensor.hit : lookAhead;
}

static bool FindClosestEnemy( gCycle * cycle, eCoord & relativePosition, REAL & enemySpeed, eCoord & enemyHeading )
{
    relativePosition = eCoord( 0, 0 );
    enemySpeed = 0;
    enemyHeading = eCoord( 0, 1 );
    REAL bestDistanceSq = 1E+30f;
    gCycle * closest = 0;
    const tList< eGameObject > & gameObjects = cycle->Grid()->GameObjects();
    for ( int i = gameObjects.Len() - 1; i >= 0; --i )
    {
        gCycle * other = dynamic_cast< gCycle * >( gameObjects( i ) );
        if ( !other || !other->Alive() || other == cycle || other->Team() == cycle->Team() ) continue;
        eCoord delta = other->Position() - cycle->Position();
        REAL distanceSq = delta.NormSquared();
        if ( distanceSq < bestDistanceSq ) { bestDistanceSq = distanceSq; closest = other; }
    }
    if ( !closest ) return false;
    eCoord delta = closest->Position() - cycle->Position();
    relativePosition = delta.Turn( cycle->Direction().Conj() ).Turn( 0, 1 );
    enemySpeed = closest->Speed();
    enemyHeading = NormalizedDir( closest->Direction() ).Turn( cycle->Direction().Conj() ).Turn( 0, 1 );
    return true;
}

class gTrainedAI : public gSimpleAI
{
public:
    gTrainedAI()
        : policyView_( Policy::Get().AcquireView() ),
          trainThisEpisode_( Policy::Get().IsLiveView( policyView_ ) ),
          hasEnemyHistory_( false ),
          prevEnemyDistance_( 0 ),
          prevEnemySide_( 0 ),
          historyReady_( false )
    {
        episode_.reserve( 512 );
        ResetTemporalState();
    }

    virtual void OnRoundResult( bool survived, REAL distance ) override
    {
        if ( distance < 0 ) distance = 0;
        REAL reward = distance * sg_rewardDistance + ( survived ? sg_rewardWin : sg_rewardDeath );
        REAL averagePredictedValue = AveragePredictedValue( episode_ );
        RecordEpisode( episode_, survived, distance, reward );
        bool learned = false;
        if ( sg_learn && trainThisEpisode_ )
        {
            Policy::Get().Train( episode_, reward );
            learned = true;
        }
        TrainingMetrics::Get().RecordEpisode(
            survived,
            distance,
            reward,
            averagePredictedValue,
            static_cast< unsigned int >( episode_.size() ),
            learned,
            Policy::Get().Episodes(),
            Policy::Get().Updates() );
        episode_.clear();
        policyView_ = Policy::Get().AcquireView();
        trainThisEpisode_ = Policy::Get().IsLiveView( policyView_ );
        ResetTemporalState();
    }

protected:
    virtual REAL DoThink() override
    {
        gCycle * cycle = Object();
        if ( !cycle || !cycle->Alive() ) return sg_thinkTime;

        REAL speed = cycle->Speed();
        if ( speed < .1f ) speed = .1f;
        REAL lookAhead = speed * Clamp( sg_lookAheadSeconds, .2f, 20.0f );
        if ( lookAhead < 8.0f ) lookAhead = 8.0f;
        bool canLeft = cycle->CanMakeTurn( -1 );
        bool canRight = cycle->CanMakeTurn( 1 );

        eCoord dir = cycle->Direction();
        gSensorWallType frontWallType = gSENSOR_NONE;
        gSensorWallType leftWallType = gSENSOR_NONE;
        gSensorWallType rightWallType = gSENSOR_NONE;
        REAL frontDistance = SenseDistance( cycle, dir, lookAhead, &frontWallType );
        REAL frontNarrowLeftDistance = SenseDistance( cycle, dir.Turn( .923879533f, .382683432f ), lookAhead );
        REAL frontNarrowRightDistance = SenseDistance( cycle, dir.Turn( .923879533f, -.382683432f ), lookAhead );
        REAL frontLeftDistance = SenseDistance( cycle, dir.Turn( .70710678f, .70710678f ), lookAhead );
        REAL frontRightDistance = SenseDistance( cycle, dir.Turn( .70710678f, -.70710678f ), lookAhead );
        REAL wideLeftDistance = SenseDistance( cycle, dir.Turn( .382683432f, .923879533f ), lookAhead );
        REAL wideRightDistance = SenseDistance( cycle, dir.Turn( .382683432f, -.923879533f ), lookAhead );
        REAL leftDistance = SenseDistance( cycle, dir.Turn( 0, 1 ), lookAhead, &leftWallType );
        REAL rightDistance = SenseDistance( cycle, dir.Turn( 0, -1 ), lookAhead, &rightWallType );
        REAL backLeftDistance = SenseDistance( cycle, dir.Turn( -.70710678f, .70710678f ), lookAhead );
        REAL backRightDistance = SenseDistance( cycle, dir.Turn( -.70710678f, -.70710678f ), lookAhead );
        REAL backDistance = SenseDistance( cycle, dir.Turn( -1, 0 ), lookAhead );

        REAL frame[kBaseFeatures] = { 0 };
        frame[kBias] = 1.0f;
        frame[kFront] = Clamp( frontDistance / lookAhead, 0.0f, 1.0f );
        frame[kFrontNarrowLeft] = Clamp( frontNarrowLeftDistance / lookAhead, 0.0f, 1.0f );
        frame[kFrontNarrowRight] = Clamp( frontNarrowRightDistance / lookAhead, 0.0f, 1.0f );
        frame[kFrontLeft] = Clamp( frontLeftDistance / lookAhead, 0.0f, 1.0f );
        frame[kFrontRight] = Clamp( frontRightDistance / lookAhead, 0.0f, 1.0f );
        frame[kWideLeft] = Clamp( wideLeftDistance / lookAhead, 0.0f, 1.0f );
        frame[kWideRight] = Clamp( wideRightDistance / lookAhead, 0.0f, 1.0f );
        frame[kLeft] = Clamp( leftDistance / lookAhead, 0.0f, 1.0f );
        frame[kRight] = Clamp( rightDistance / lookAhead, 0.0f, 1.0f );
        frame[kBackLeft] = Clamp( backLeftDistance / lookAhead, 0.0f, 1.0f );
        frame[kBackRight] = Clamp( backRightDistance / lookAhead, 0.0f, 1.0f );
        frame[kBack] = Clamp( backDistance / lookAhead, 0.0f, 1.0f );
        frame[kSpeed] = speed / ( speed + 20.0f );
        frame[kCanLeft] = canLeft ? 1.0f : 0.0f;
        frame[kCanRight] = canRight ? 1.0f : 0.0f;
        frame[kTurnDelay] = Clamp( cycle->GetTurnDelay() / ( sg_lookAheadSeconds + .1f ), 0.0f, 1.0f );

        eCoord enemyPos;
        REAL enemySpeed = 0;
        eCoord enemyHeading;
        if ( FindClosestEnemy( cycle, enemyPos, enemySpeed, enemyHeading ) )
        {
            REAL dist = std::sqrt( enemyPos.NormSquared() );
            REAL denom = dist + 1.0f;
            frame[kEnemyAhead] = Clamp( enemyPos.y / denom, -1.0f, 1.0f );
            frame[kEnemySide] = Clamp( enemyPos.x / denom, -1.0f, 1.0f );
            frame[kEnemyNear] = 1.0f - Clamp( dist / ( lookAhead * 2.0f ), 0.0f, 1.0f );
            frame[kEnemySpeedDiff] = ( enemySpeed - speed ) / ( std::fabs( enemySpeed ) + std::fabs( speed ) + 1.0f );
            frame[kEnemyHeadingDot] = Clamp( enemyHeading.y, -1.0f, 1.0f );
            frame[kEnemyCrossing] = Clamp( enemyHeading.x, -1.0f, 1.0f );
            frame[kEnemyFrontProximity] = frame[kEnemyNear] * Clamp( frame[kEnemyAhead], 0.0f, 1.0f );

            if ( hasEnemyHistory_ )
            {
                REAL denomDistance = lookAhead + 1.0f;
                frame[kEnemyClosing] = Clamp( ( prevEnemyDistance_ - dist ) / denomDistance, -1.0f, 1.0f );
                frame[kEnemyLateralClosing] = Clamp( ( enemyPos.x - prevEnemySide_ ) / denomDistance, -1.0f, 1.0f );
                REAL prevSideNormalized = Clamp( prevEnemySide_ / ( prevEnemyDistance_ + 1.0f ), -1.0f, 1.0f );
                frame[kEnemyBearingDrift] = Clamp( frame[kEnemySide] - prevSideNormalized, -1.0f, 1.0f );
            }
            prevEnemyDistance_ = dist;
            prevEnemySide_ = enemyPos.x;
            hasEnemyHistory_ = true;
        }
        else
        {
            hasEnemyHistory_ = false;
        }
        frame[kFrontEnemyWall] = frontWallType == gSENSOR_ENEMY ? 1.0f : 0.0f;
        frame[kFrontRimWall] = frontWallType == gSENSOR_RIM ? 1.0f : 0.0f;
        frame[kLeftEnemyWall] = leftWallType == gSENSOR_ENEMY ? 1.0f : 0.0f;
        frame[kRightEnemyWall] = rightWallType == gSENSOR_ENEMY ? 1.0f : 0.0f;
        frame[kLeftRimWall] = leftWallType == gSENSOR_RIM ? 1.0f : 0.0f;
        frame[kRightRimWall] = rightWallType == gSENSOR_RIM ? 1.0f : 0.0f;
        frame[kLeftRightBalance] = Clamp( frame[kLeft] - frame[kRight], -1.0f, 1.0f );
        frame[kFrontPressure] = 1.0f - frame[kFront];
        frame[kBackPressure] = 1.0f - frame[kBack];
        frame[kWallCrowding] = 1.0f - Clamp(
            ( frame[kFront] + frame[kFrontNarrowLeft] + frame[kFrontNarrowRight] + frame[kLeft] + frame[kRight] + frame[kBack] ) / 6.0f,
            0.0f,
            1.0f );
        frame[kForwardArcSafety] = Clamp(
            ( frame[kFront] + frame[kFrontNarrowLeft] + frame[kFrontNarrowRight] + frame[kFrontLeft] + frame[kFrontRight] + frame[kWideLeft] + frame[kWideRight] ) / 7.0f,
            0.0f,
            1.0f );
        frame[kSideArcSafety] = Clamp(
            ( frame[kLeft] + frame[kRight] + frame[kWideLeft] + frame[kWideRight] + frame[kBackLeft] + frame[kBackRight] ) / 6.0f,
            0.0f,
            1.0f );
        frame[kEnemyBackProximity] = frame[kEnemyNear] * Clamp( -frame[kEnemyAhead], 0.0f, 1.0f );
        frame[kSpeedPressure] = Clamp( frame[kSpeed] * frame[kFrontPressure], 0.0f, 1.0f );
        frame[kEscapeLeft] = Max4( frame[kWideLeft], frame[kLeft], frame[kFrontLeft], frame[kBackLeft] );
        frame[kEscapeRight] = Max4( frame[kWideRight], frame[kRight], frame[kFrontRight], frame[kBackRight] );
        frame[kEscapeRouteBias] = Clamp(
            frame[kEscapeLeft] - frame[kEscapeRight],
            -1.0f,
            1.0f );

        REAL x[kFeatures] = { 0 };
        BuildStackedFeatures( frame, x );

        REAL h0[kTemporalHidden];
        REAL h1[kHidden1];
        REAL h2[kHidden2];
        REAL h3[kHidden3];
        REAL p[kActions];
        REAL v = 0;
        int action = Policy::Get().Choose( policyView_, x, canLeft, canRight, h0, h1, h2, h3, p, v );
        int turn = ActionToTurn( action );
        if ( turn != 0 && cycle->CanMakeTurn( turn ) ) cycle->Turn( turn );

        if ( ( sg_learn || sg_record ) && static_cast< int >( episode_.size() ) < sg_maxEpisodeSteps )
        {
            Step s;
            for ( int i = 0; i < kFeatures; ++i ) s.x[i] = x[i];
            for ( int i = 0; i < kTemporalHidden; ++i ) s.h0[i] = h0[i];
            for ( int i = 0; i < kHidden1; ++i ) s.h1[i] = h1[i];
            for ( int i = 0; i < kHidden2; ++i ) s.h2[i] = h2[i];
            for ( int i = 0; i < kHidden3; ++i ) s.h3[i] = h3[i];
            for ( int i = 0; i < kActions; ++i ) s.p[i] = p[i];
            s.v = v;
            s.action = action;
            s.canLeft = canLeft;
            s.canRight = canRight;
            episode_.push_back( s );
        }

        return Clamp( sg_thinkTime, 0.02f, 1.0f );
    }

private:
    void ResetTemporalState()
    {
        hasEnemyHistory_ = false;
        prevEnemyDistance_ = 0;
        prevEnemySide_ = 0;
        historyReady_ = false;
        for ( int frame = 0; frame < kHistoryFrames; ++frame )
            for ( int feature = 0; feature < kBaseFeatures; ++feature )
                history_[frame][feature] = 0;
    }

    void BuildStackedFeatures( REAL const current[kBaseFeatures], REAL stacked[kFeatures] )
    {
        if ( !historyReady_ )
        {
            for ( int frame = 0; frame < kHistoryFrames; ++frame )
                for ( int feature = 0; feature < kBaseFeatures; ++feature )
                    history_[frame][feature] = current[feature];
            historyReady_ = true;
        }
        else
        {
            for ( int frame = 0; frame < kHistoryFrames - 1; ++frame )
                for ( int feature = 0; feature < kBaseFeatures; ++feature )
                    history_[frame][feature] = history_[frame + 1][feature];

            for ( int feature = 0; feature < kBaseFeatures; ++feature )
                history_[kHistoryFrames - 1][feature] = current[feature];
        }

        for ( int frame = 0; frame < kHistoryFrames; ++frame )
            for ( int feature = 0; feature < kBaseFeatures; ++feature )
                stacked[frame * kBaseFeatures + feature] = history_[frame][feature];
    }

    std::vector< Step > episode_;
    int policyView_;
    bool trainThisEpisode_;
    bool hasEnemyHistory_;
    REAL prevEnemyDistance_;
    REAL prevEnemySide_;
    bool historyReady_;
    REAL history_[kHistoryFrames][kBaseFeatures];
};

class gTrainedAIFactory : public gSimpleAIFactory
{
protected:
    virtual gSimpleAI * DoCreate() const override { return tNEW( gTrainedAI )(); }
};

static gTrainedAIFactory sg_factory;

static void ApplyFactorySetting()
{
    if ( sg_enable )
    {
        if ( gSimpleAIFactory::Get() != &sg_factory )
        {
            gSimpleAIFactory::Set( &sg_factory );
            con << sg_aiName << ": neural policy enabled.\n";
        }
    }
    else if ( gSimpleAIFactory::Get() == &sg_factory )
    {
        gSimpleAIFactory::Set( NULL );
        con << sg_aiName << ": neural policy disabled.\n";
    }
}

static void OnEnableChanged()
{
    ApplyFactorySetting();
}

static tConfItem< bool > sg_enableConf( "AI_TRAINED_ENABLE", sg_enable, &OnEnableChanged );
static tConfItem< bool > sg_learnConf( "AI_TRAINED_LEARN", sg_learn );
static tConfItem< bool > sg_recordConf( "AI_TRAINED_RECORD", sg_record );
static tConfItem< bool > sg_autostartConf( "AI_TRAINED_AUTOSTART", sg_autostart );
static tConfItem< int > sg_botCountConf( "AI_TRAINED_BOT_COUNT", sg_botCount );
static tConfItem< tString > sg_modelFileConf( "AI_TRAINED_MODEL_FILE", sg_modelFile );
static tConfItem< tString > sg_recordFileConf( "AI_TRAINED_RECORD_FILE", sg_recordFile );
static tConfItem< tString > sg_metricsFileConf( "AI_TRAINED_METRICS_FILE", sg_metricsFile );
static tConfItem< tString > sg_checkpointPrefixConf( "AI_TRAINED_CHECKPOINT_PREFIX", sg_checkpointPrefix );
static tConfItem< REAL > sg_thinkTimeConf( "AI_TRAINED_THINK_TIME", sg_thinkTime );
static tConfItem< REAL > sg_lookAheadConf( "AI_TRAINED_LOOKAHEAD_SECONDS", sg_lookAheadSeconds );
static tConfItem< REAL > sg_explorationConf( "AI_TRAINED_EXPLORATION", sg_exploration );
static tConfItem< REAL > sg_learningRateConf( "AI_TRAINED_LEARNING_RATE", sg_learningRate );
static tConfItem< REAL > sg_valueLearningRateConf( "AI_TRAINED_VALUE_LEARNING_RATE", sg_valueLearningRate );
static tConfItem< REAL > sg_baselineDecayConf( "AI_TRAINED_BASELINE_DECAY", sg_baselineDecay );
static tConfItem< REAL > sg_discountConf( "AI_TRAINED_DISCOUNT", sg_discount );
static tConfItem< REAL > sg_weightDecayConf( "AI_TRAINED_WEIGHT_DECAY", sg_weightDecay );
static tConfItem< REAL > sg_weightClipConf( "AI_TRAINED_WEIGHT_CLIP", sg_weightClip );
static tConfItem< int > sg_saveEveryConf( "AI_TRAINED_SAVE_EVERY", sg_saveEvery );
static tConfItem< int > sg_checkpointEveryConf( "AI_TRAINED_CHECKPOINT_EVERY", sg_checkpointEvery );
static tConfItem< int > sg_maxEpisodeConf( "AI_TRAINED_MAX_EPISODE_STEPS", sg_maxEpisodeSteps );
static tConfItem< int > sg_trainEpochsConf( "AI_TRAINED_TRAIN_EPOCHS", sg_trainEpochs );
static tConfItem< int > sg_recordStrideConf( "AI_TRAINED_RECORD_STRIDE", sg_recordStride );
static tConfItem< int > sg_policyPoolSizeConf( "AI_TRAINED_POLICY_POOL_SIZE", sg_policyPoolSize );
static tConfItem< int > sg_policySnapshotEveryConf( "AI_TRAINED_POLICY_SNAPSHOT_EVERY", sg_policySnapshotEvery );
static tConfItem< int > sg_policySnapshotWarmupConf( "AI_TRAINED_POLICY_SNAPSHOT_WARMUP", sg_policySnapshotWarmup );
static tConfItem< REAL > sg_policyHistoricProbConf( "AI_TRAINED_POLICY_HISTORIC_PROB", sg_policyHistoricProb );
static tConfItem< REAL > sg_rewardDistanceConf( "AI_TRAINED_REWARD_DISTANCE", sg_rewardDistance );
static tConfItem< REAL > sg_rewardWinConf( "AI_TRAINED_REWARD_WIN", sg_rewardWin );
static tConfItem< REAL > sg_rewardDeathConf( "AI_TRAINED_REWARD_DEATH", sg_rewardDeath );
static tConfItem< bool > sg_offlineTrainConf( "AI_TRAINED_OFFLINE_TRAIN", sg_offlineTrain );
static tConfItem< tString > sg_offlineSourceListConf( "AI_TRAINED_OFFLINE_SOURCE_LIST", sg_offlineSourceList );
static tConfItem< tString > sg_offlineStateFileConf( "AI_TRAINED_OFFLINE_STATE_FILE", sg_offlineStateFile );
}

bool & gTrainedAI_Enable() { return sg_enable; }
bool & gTrainedAI_Learn() { return sg_learn; }
bool & gTrainedAI_Record() { return sg_record; }
bool & gTrainedAI_Autostart() { return sg_autostart; }
int & gTrainedAI_BotCount() { return sg_botCount; }
char const * gTrainedAI_Name() { return sg_aiName; }

void gTrainedAI_InstallFactoryIfEnabled()
{
    RunOfflineTrainerIfConfigured();
    ApplyFactorySetting();
}
