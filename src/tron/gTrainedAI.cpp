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
#include "gWall.h"
#include "tConfiguration.h"
#include "tConsole.h"
#include "tDirectories.h"
#include "tRandom.h"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace
{
enum
{
    kTeacherScalarFeatures = 6,
    kTeacherMapSize = 25,
    kTeacherMapChannels = 7,
    kTeacherMapCells = kTeacherMapSize * kTeacherMapSize,
    kFeatures = kTeacherScalarFeatures + kTeacherMapSize * kTeacherMapSize * kTeacherMapChannels,
    kConv1Channels = 32,
    kConv2Channels = 64,
    kConvKernelSize = 3,
    kDenseInput = kTeacherScalarFeatures + kTeacherMapCells * kConv2Channels,
    kHidden1 = 256,
    kHidden2 = 128,
    kActions = 3
};

enum
{
    kParameterCount =
        kConv1Channels * kTeacherMapChannels * kConvKernelSize * kConvKernelSize + kConv1Channels +
        kConv2Channels * kConv1Channels * kConvKernelSize * kConvKernelSize + kConv2Channels +
        kDenseInput * kHidden1 + kHidden1 +
        kHidden1 * kHidden2 + kHidden2 +
        kActions * kHidden2 + kActions +
        kHidden2 + 1
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
static tString sg_modelFile( "trained_ai_teacher_cnn_model.txt" );
static tString sg_recordFile( "trained_ai_experience.log" );
static tString sg_teacherFile( "trained_ai_teacher_examples.log" );
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

enum TeacherMapChannel
{
    kTeacherSelfWall = 0,
    kTeacherTeamWall,
    kTeacherEnemyWall,
    kTeacherRimWall,
    kTeacherSelfCycle,
    kTeacherTeamCycle,
    kTeacherEnemyCycle
};

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

static int TurnToAction( int turn )
{
    if ( turn < 0 ) return 0;
    if ( turn > 0 ) return 2;
    return 1;
}

static REAL ComputeLookAhead( gCycle * cycle )
{
    REAL speed = cycle ? cycle->Speed() : 0;
    if ( speed < .1f ) speed = .1f;
    REAL lookAhead = speed * Clamp( sg_lookAheadSeconds, .2f, 20.0f );
    if ( lookAhead < 8.0f ) lookAhead = 8.0f;
    return lookAhead;
}

static eCoord ToCycleLocalSpace( gCycle * cycle, eCoord const & worldPoint )
{
    if ( !cycle )
    {
        return eCoord( 0, 0 );
    }

    return ( worldPoint - cycle->Position() ).Turn( cycle->Direction().Conj() ).Turn( 0, 1 );
}

struct Step
{
    REAL x[kFeatures];
    REAL p[kActions];
    REAL v;
    int action;
    bool canLeft;
    bool canRight;
};

struct LearningStats
{
    LearningStats()
        : policyLoss( 0 ),
          valueLoss( 0 ),
          entropy( 0 ),
          steps( 0 ),
          valid( false )
    {
    }

    REAL policyLoss;
    REAL valueLoss;
    REAL entropy;
    unsigned int steps;
    bool valid;
};

struct TeacherEpisodeState
{
    TeacherEpisodeState()
        : episodeId( 0 ),
          stepIndex( 0 )
    {
    }

    unsigned long long episodeId;
    unsigned int stepIndex;
};

struct TeacherExample
{
    TeacherExample()
        : action( 1 ),
          canLeft( false ),
          canRight( false ),
          halfExtent( 0 )
    {
        for ( int i = 0; i < kTeacherScalarFeatures; ++i )
        {
            scalars[i] = 0;
        }
        for ( int channel = 0; channel < kTeacherMapChannels; ++channel )
            for ( int y = 0; y < kTeacherMapSize; ++y )
                for ( int x = 0; x < kTeacherMapSize; ++x )
                    map[channel][y][x] = 0;
    }

    int action;
    bool canLeft;
    bool canRight;
    REAL halfExtent;
    REAL scalars[kTeacherScalarFeatures];
    REAL map[kTeacherMapChannels][kTeacherMapSize][kTeacherMapSize];
};

static unsigned long long sg_teacherEpisodeId = 0;
static std::map< gCycle const *, TeacherEpisodeState > sg_teacherEpisodes;
static std::set< eWall * > * sg_teacherWallCollector = 0;

static void CollectTeacherWall( eWall * wall )
{
    if ( sg_teacherWallCollector && wall )
    {
        sg_teacherWallCollector->insert( wall );
    }
}

static void MarkTeacherCell( TeacherExample & example, int channel, int x, int y, REAL value = 1.0f )
{
    if ( channel < 0 || channel >= kTeacherMapChannels ) return;
    if ( x < 0 || x >= kTeacherMapSize ) return;
    if ( y < 0 || y >= kTeacherMapSize ) return;
    if ( value > example.map[channel][y][x] )
    {
        example.map[channel][y][x] = value;
    }
}

static bool LocalToTeacherCell( eCoord const & local, REAL halfExtent, int & x, int & y )
{
    if ( halfExtent <= 0 )
    {
        return false;
    }

    REAL cellSize = ( halfExtent * 2.0f ) / static_cast< REAL >( kTeacherMapSize );
    if ( cellSize <= 0 )
    {
        return false;
    }

    REAL fx = ( local.x + halfExtent ) / cellSize;
    REAL fy = ( halfExtent - local.y ) / cellSize;
    x = static_cast< int >( std::floor( fx ) );
    y = static_cast< int >( std::floor( fy ) );
    return x >= 0 && x < kTeacherMapSize && y >= 0 && y < kTeacherMapSize;
}

static void RasterizeTeacherSegment( TeacherExample & example, int channel, eCoord const & localStart, eCoord const & localEnd )
{
    REAL span = std::max( std::fabs( localEnd.x - localStart.x ), std::fabs( localEnd.y - localStart.y ) );
    REAL cellSize = ( example.halfExtent * 2.0f ) / static_cast< REAL >( kTeacherMapSize );
    int steps = cellSize > 0 ? static_cast< int >( std::ceil( span / cellSize ) ) : 1;
    if ( steps < 1 ) steps = 1;
    steps *= 2;

    for ( int i = 0; i <= steps; ++i )
    {
        REAL t = static_cast< REAL >( i ) / static_cast< REAL >( steps );
        eCoord local = localStart * ( 1.0f - t ) + localEnd * t;
        int x = 0;
        int y = 0;
        if ( LocalToTeacherCell( local, example.halfExtent, x, y ) )
        {
            MarkTeacherCell( example, channel, x, y );
        }
    }
}

static TeacherEpisodeState & GetTeacherEpisodeState( gCycle * cycle )
{
    TeacherEpisodeState & state = sg_teacherEpisodes[cycle];
    if ( state.episodeId == 0 )
    {
        state.episodeId = ++sg_teacherEpisodeId;
        state.stepIndex = 0;
    }
    return state;
}

static void BuildTeacherExample( gCycle * cycle, int turn, TeacherExample & example )
{
    if ( !cycle ) return;

    REAL speed = cycle->Speed();
    if ( speed < .1f ) speed = .1f;
    REAL lookAhead = ComputeLookAhead( cycle );
    REAL halfExtent = Clamp( lookAhead * 1.5f, 18.0f, 80.0f );
    bool canLeft = cycle->CanMakeTurn( -1 );
    bool canRight = cycle->CanMakeTurn( 1 );

    example.action = TurnToAction( turn );
    example.canLeft = canLeft;
    example.canRight = canRight;
    example.halfExtent = halfExtent;
    example.scalars[0] = speed / ( speed + 20.0f );
    example.scalars[1] = Clamp( cycle->GetTurnDelay() / ( sg_lookAheadSeconds + .1f ), 0.0f, 1.0f );
    example.scalars[2] = canLeft ? 1.0f : 0.0f;
    example.scalars[3] = canRight ? 1.0f : 0.0f;
    example.scalars[4] = Clamp( lookAhead / ( lookAhead + 20.0f ), 0.0f, 1.0f );
    example.scalars[5] = Clamp( halfExtent / ( halfExtent + 40.0f ), 0.0f, 1.0f );

    MarkTeacherCell( example, kTeacherSelfCycle, kTeacherMapSize / 2, kTeacherMapSize / 2 );

    if ( cycle->Grid() )
    {
        std::set< eWall * > nearbyWalls;
        sg_teacherWallCollector = &nearbyWalls;
        cycle->Grid()->ProcessWallsInRange(
            &CollectTeacherWall,
            cycle->Position(),
            halfExtent * 1.5f,
            cycle->CurrentFace() );
        sg_teacherWallCollector = 0;

        for ( std::set< eWall * >::const_iterator it = nearbyWalls.begin(); it != nearbyWalls.end(); ++it )
        {
            eWall * wall = *it;
            if ( !wall ) continue;

            int channel = kTeacherRimWall;
            if ( gPlayerWall * playerWall = dynamic_cast< gPlayerWall * >( wall ) )
            {
                gCycle * owner = playerWall->Cycle();
                if ( owner && owner == cycle )
                {
                    channel = kTeacherSelfWall;
                }
                else if ( owner && owner->Team() == cycle->Team() )
                {
                    channel = kTeacherTeamWall;
                }
                else
                {
                    channel = kTeacherEnemyWall;
                }
            }

            eCoord localStart = ToCycleLocalSpace( cycle, wall->EndPoint( 0 ) );
            eCoord localEnd = ToCycleLocalSpace( cycle, wall->EndPoint( 1 ) );
            RasterizeTeacherSegment( example, channel, localStart, localEnd );
        }

        const tList< eGameObject > & gameObjects = cycle->Grid()->GameObjects();
        for ( int i = gameObjects.Len() - 1; i >= 0; --i )
        {
            gCycle * other = dynamic_cast< gCycle * >( gameObjects( i ) );
            if ( !other || !other->Alive() ) continue;

            int channel = kTeacherEnemyCycle;
            if ( other == cycle )
            {
                channel = kTeacherSelfCycle;
            }
            else if ( other->Team() == cycle->Team() )
            {
                channel = kTeacherTeamCycle;
            }

            int x = 0;
            int y = 0;
            if ( LocalToTeacherCell( ToCycleLocalSpace( cycle, other->Position() ), halfExtent, x, y ) )
            {
                MarkTeacherCell( example, channel, x, y );
            }
        }
    }
}

static void FlattenTeacherExample( TeacherExample const & example, REAL x[kFeatures] )
{
    int index = 0;
    for ( int i = 0; i < kTeacherScalarFeatures; ++i )
    {
        x[index++] = example.scalars[i];
    }

    for ( int channel = 0; channel < kTeacherMapChannels; ++channel )
        for ( int y = 0; y < kTeacherMapSize; ++y )
            for ( int cellX = 0; cellX < kTeacherMapSize; ++cellX )
                x[index++] = example.map[channel][y][cellX];
}

static void UnpackTeacherInput( REAL const x[kFeatures], REAL scalars[kTeacherScalarFeatures], REAL map[kTeacherMapChannels][kTeacherMapSize][kTeacherMapSize] )
{
    int index = 0;
    for ( int i = 0; i < kTeacherScalarFeatures; ++i )
    {
        scalars[i] = x[index++];
    }

    for ( int channel = 0; channel < kTeacherMapChannels; ++channel )
        for ( int y = 0; y < kTeacherMapSize; ++y )
            for ( int cellX = 0; cellX < kTeacherMapSize; ++cellX )
                map[channel][y][cellX] = x[index++];
}

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

    void RecordEpisode( bool survived, REAL distance, REAL reward, REAL averagePredictedValue, unsigned int steps, LearningStats const & learningStats, bool learned, unsigned int policyEpisodes, unsigned int policyUpdates )
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
        if ( learningStats.valid )
        {
            ++learnedEpisodes_;
            policyLossTotal_ += learningStats.policyLoss;
            valueLossTotal_ += learningStats.valueLoss;
            entropyTotal_ += learningStats.entropy;
            lastPolicyLoss_ = learningStats.policyLoss;
            lastValueLoss_ = learningStats.valueLoss;
            lastEntropy_ = learningStats.entropy;
        }

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
            out << "episode,survived,reward,distance,average_predicted_value,steps,learned,policy_episodes,policy_updates,cumulative_win_rate,cumulative_average_reward,cumulative_average_distance,cumulative_average_predicted_value,policy_loss,value_loss,entropy,cumulative_average_policy_loss,cumulative_average_value_loss,cumulative_average_entropy\n";
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
            << "," << ( learningStats.valid ? learningStats.policyLoss : 0 )
            << "," << ( learningStats.valid ? learningStats.valueLoss : 0 )
            << "," << ( learningStats.valid ? learningStats.entropy : 0 )
            << "," << AveragePolicyLoss()
            << "," << AverageValueLoss()
            << "," << AverageEntropy()
            << "\n";

        WriteSummary( survived, distance, reward, averagePredictedValue, steps, learningStats, learned, policyEpisodes, policyUpdates );
    }

private:
    TrainingMetrics()
        : episodes_( 0 ),
          wins_( 0 ),
          learnedEpisodes_( 0 ),
          rewardTotal_( 0 ),
          distanceTotal_( 0 ),
          predictedValueTotal_( 0 ),
          stepsTotal_( 0 ),
          policyLossTotal_( 0 ),
          valueLossTotal_( 0 ),
          entropyTotal_( 0 ),
          lastPolicyLoss_( 0 ),
          lastValueLoss_( 0 ),
          lastEntropy_( 0 ),
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

    REAL AveragePolicyLoss() const
    {
        if ( learnedEpisodes_ <= 0 )
        {
            return 0;
        }

        return policyLossTotal_ / static_cast< REAL >( learnedEpisodes_ );
    }

    REAL AverageValueLoss() const
    {
        if ( learnedEpisodes_ <= 0 )
        {
            return 0;
        }

        return valueLossTotal_ / static_cast< REAL >( learnedEpisodes_ );
    }

    REAL AverageEntropy() const
    {
        if ( learnedEpisodes_ <= 0 )
        {
            return 0;
        }

        return entropyTotal_ / static_cast< REAL >( learnedEpisodes_ );
    }

    void WriteSummary( bool survived, REAL distance, REAL reward, REAL averagePredictedValue, unsigned int steps, LearningStats const & learningStats, bool learned, unsigned int policyEpisodes, unsigned int policyUpdates )
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
        out << "steps_total " << stepsTotal_ << "\n";
        out << "learned_episodes " << learnedEpisodes_ << "\n";
        out << "average_policy_loss " << AveragePolicyLoss() << "\n";
        out << "average_value_loss " << AverageValueLoss() << "\n";
        out << "average_entropy " << AverageEntropy() << "\n";
        out << "last_survived " << ( survived ? 1 : 0 ) << "\n";
        out << "last_reward " << reward << "\n";
        out << "last_distance " << distance << "\n";
        out << "last_average_predicted_value " << averagePredictedValue << "\n";
        out << "last_steps " << steps << "\n";
        out << "last_policy_loss " << ( learningStats.valid ? learningStats.policyLoss : lastPolicyLoss_ ) << "\n";
        out << "last_value_loss " << ( learningStats.valid ? learningStats.valueLoss : lastValueLoss_ ) << "\n";
        out << "last_entropy " << ( learningStats.valid ? learningStats.entropy : lastEntropy_ ) << "\n";
        out << "last_learned " << ( learned ? 1 : 0 ) << "\n";
        out << "policy_episodes " << policyEpisodes << "\n";
        out << "policy_updates " << policyUpdates << "\n";
    }

    unsigned long long episodes_;
    unsigned long long wins_;
    unsigned long long learnedEpisodes_;
    REAL rewardTotal_;
    REAL distanceTotal_;
    REAL predictedValueTotal_;
    unsigned long long stepsTotal_;
    REAL policyLossTotal_;
    REAL valueLossTotal_;
    REAL entropyTotal_;
    REAL lastPolicyLoss_;
    REAL lastValueLoss_;
    REAL lastEntropy_;
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

    int Choose( int viewIndex, REAL const x[kFeatures], bool canLeft, bool canRight, REAL p[kActions], REAL & v )
    {
        EnsureLoaded();

        ForwardCache cache;
        Forward( GetView( viewIndex ), x, canLeft, canRight, cache );
        for ( int a = 0; a < kActions; ++a )
        {
            p[a] = cache.p[a];
        }
        v = cache.v;

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

    LearningStats Train( std::vector< Step > const & episode, REAL reward )
    {
        LearningStats stats;
        EnsureLoaded();
        if ( episode.empty() ) return stats;

        REAL lr = Clamp( sg_learningRate, 0.0f, 1.0f );
        if ( lr <= 0 ) return stats;

        REAL baselineReward = baseline_;
        REAL bDecay = Clamp( sg_baselineDecay, 0.0001f, 1.0f );
        baseline_ = baseline_ * ( 1.0f - bDecay ) + reward * bDecay;

        int steps = static_cast< int >( episode.size() );
        std::vector< REAL > returns( steps, 0 );
        std::vector< REAL > advantages( steps, 0 );
        REAL discount = Clamp( sg_discount, 0.0f, 1.0f );
        REAL discountedReturn = reward;
        REAL discountedBaseline = baselineReward;
        for ( int t = steps - 1; t >= 0; --t )
        {
            returns[t] = discountedReturn;
            advantages[t] = returns[t] - discountedBaseline;
            discountedReturn *= discount;
            discountedBaseline *= discount;
        }

        REAL meanSquare = 0;
        for ( int t = 0; t < steps; ++t )
        {
            meanSquare += advantages[t] * advantages[t];
        }
        meanSquare /= static_cast< REAL >( steps );
        REAL invRms = meanSquare > 1E-8f ? 1.0f / std::sqrt( meanSquare + 1E-6f ) : 1.0f;
        REAL policyLossTotal = 0;
        REAL valueLossTotal = 0;
        REAL entropyTotal = 0;

        int epochs = sg_trainEpochs < 1 ? 1 : sg_trainEpochs;
        for ( int epoch = 0; epoch < epochs; ++epoch )
        {
            for ( int t = 0; t < steps; ++t )
            {
                Step const & s = episode[t];
                REAL normalizedAdvantage = Clamp( advantages[t] * invRms, -4.0f, 4.0f );
                REAL policyScale = lr * normalizedAdvantage / static_cast< REAL >( steps * epochs );

                ForwardCache cache;
                Forward( CurrentView(), s.x, s.canLeft, s.canRight, cache );

                REAL valueError = Clamp( returns[t] - cache.v, -4.0f, 4.0f );
                if ( epoch == 0 )
                {
                    REAL probability = cache.p[s.action];
                    if ( probability < 1E-6f )
                    {
                        probability = 1E-6f;
                    }

                    policyLossTotal += -normalizedAdvantage * std::log( probability );
                    valueLossTotal += 0.5f * valueError * valueError;

                    REAL stepEntropy = 0;
                    for ( int a = 0; a < kActions; ++a )
                    {
                        if ( cache.p[a] > 1E-6f )
                        {
                            stepEntropy += -cache.p[a] * std::log( cache.p[a] );
                        }
                    }
                    entropyTotal += stepEntropy;
                }

                REAL valueScale = Clamp( sg_valueLearningRate, 0.0f, 1.0f ) * valueError / static_cast< REAL >( steps * epochs );
                if ( policyScale == 0 && valueScale == 0 ) continue;

                Backward( cache, s.action, policyScale, valueScale );
            }
        }

        ApplyWeightDecay( lr );

        ++episodes_;
        ++updates_;
        dirty_ = true;
        MaybeAddSnapshot();
        MaybeWriteCheckpoint();
        if ( sg_saveEvery < 1 ) sg_saveEvery = 1;
        if ( episodes_ % static_cast< unsigned int >( sg_saveEvery ) == 0 ) Save();
        stats.policyLoss = policyLossTotal / static_cast< REAL >( steps );
        stats.valueLoss = valueLossTotal / static_cast< REAL >( steps );
        stats.entropy = entropyTotal / static_cast< REAL >( steps );
        stats.steps = static_cast< unsigned int >( steps );
        stats.valid = true;
        return stats;
    }

    LearningStats TrainTeacher( std::vector< Step > const & episode )
    {
        LearningStats stats;
        EnsureLoaded();
        if ( episode.empty() ) return stats;

        REAL lr = Clamp( sg_learningRate, 0.0f, 1.0f );
        if ( lr <= 0 ) return stats;

        REAL policyLossTotal = 0;
        REAL entropyTotal = 0;
        int steps = static_cast< int >( episode.size() );
        int epochs = sg_trainEpochs < 1 ? 1 : sg_trainEpochs;

        for ( int epoch = 0; epoch < epochs; ++epoch )
        {
            for ( int t = 0; t < steps; ++t )
            {
                Step const & s = episode[t];

                ForwardCache cache;
                Forward( CurrentView(), s.x, s.canLeft, s.canRight, cache );

                if ( epoch == 0 )
                {
                    REAL probability = cache.p[s.action];
                    if ( probability < 1E-6f )
                    {
                        probability = 1E-6f;
                    }
                    policyLossTotal += -std::log( probability );

                    REAL stepEntropy = 0;
                    for ( int a = 0; a < kActions; ++a )
                    {
                        if ( cache.p[a] > 1E-6f )
                        {
                            stepEntropy += -cache.p[a] * std::log( cache.p[a] );
                        }
                    }
                    entropyTotal += stepEntropy;
                }

                REAL scale = lr / static_cast< REAL >( steps * epochs );
                Backward( cache, s.action, scale, 0 );
            }
        }

        ApplyWeightDecay( lr );

        ++episodes_;
        ++updates_;
        dirty_ = true;
        MaybeWriteCheckpoint();
        if ( sg_saveEvery < 1 ) sg_saveEvery = 1;
        if ( episodes_ % static_cast< unsigned int >( sg_saveEvery ) == 0 ) Save();

        stats.policyLoss = policyLossTotal / static_cast< REAL >( steps );
        stats.valueLoss = 0;
        stats.entropy = entropyTotal / static_cast< REAL >( steps );
        stats.steps = static_cast< unsigned int >( steps );
        stats.valid = true;
        return stats;
    }

private:
    struct ForwardCache
    {
        REAL scalars[kTeacherScalarFeatures];
        REAL mapIn[kTeacherMapChannels][kTeacherMapSize][kTeacherMapSize];
        REAL conv1[kConv1Channels][kTeacherMapSize][kTeacherMapSize];
        REAL conv2[kConv2Channels][kTeacherMapSize][kTeacherMapSize];
        REAL denseIn[kDenseInput];
        REAL h0[kHidden1];
        REAL h1[kHidden2];
        REAL p[kActions];
        REAL v;
    };

    struct WeightsView
    {
        REAL const ( *conv1 )[kTeacherMapChannels][kConvKernelSize][kConvKernelSize];
        REAL const * bConv1;
        REAL const ( *conv2 )[kConv1Channels][kConvKernelSize][kConvKernelSize];
        REAL const * bConv2;
        REAL const ( *dense0 )[kDenseInput];
        REAL const * bDense0;
        REAL const ( *dense1 )[kHidden1];
        REAL const * bDense1;
        REAL const ( *policy )[kHidden2];
        REAL const * bPolicy;
        REAL const * value;
        REAL bValue;
    };

    struct Snapshot
    {
        REAL conv1[kConv1Channels][kTeacherMapChannels][kConvKernelSize][kConvKernelSize];
        REAL bConv1[kConv1Channels];
        REAL conv2[kConv2Channels][kConv1Channels][kConvKernelSize][kConvKernelSize];
        REAL bConv2[kConv2Channels];
        REAL dense0[kHidden1][kDenseInput];
        REAL bDense0[kHidden1];
        REAL dense1[kHidden2][kHidden1];
        REAL bDense1[kHidden2];
        REAL policy[kActions][kHidden2];
        REAL bPolicy[kActions];
        REAL value[kHidden2];
        REAL bValue;
        unsigned int episode;
    };

    Policy(): baseline_( 0 ), episodes_( 0 ), updates_( 0 ), loaded_( false ), dirty_( false )
    {
        ZeroWeights();
    }

    static REAL ActivationDerivative( REAL value )
    {
        return 1.0f - value * value;
    }

    void ZeroWeights()
    {
        for ( int oc = 0; oc < kConv1Channels; ++oc )
        {
            bConv1_[oc] = 0;
            for ( int ic = 0; ic < kTeacherMapChannels; ++ic )
                for ( int ky = 0; ky < kConvKernelSize; ++ky )
                    for ( int kx = 0; kx < kConvKernelSize; ++kx )
                        conv1_[oc][ic][ky][kx] = 0;
        }

        for ( int oc = 0; oc < kConv2Channels; ++oc )
        {
            bConv2_[oc] = 0;
            for ( int ic = 0; ic < kConv1Channels; ++ic )
                for ( int ky = 0; ky < kConvKernelSize; ++ky )
                    for ( int kx = 0; kx < kConvKernelSize; ++kx )
                        conv2_[oc][ic][ky][kx] = 0;
        }

        for ( int j = 0; j < kHidden1; ++j )
        {
            bDense0_[j] = 0;
            for ( int i = 0; i < kDenseInput; ++i ) dense0_[j][i] = 0;
        }
        for ( int j = 0; j < kHidden2; ++j )
        {
            bDense1_[j] = 0;
            for ( int i = 0; i < kHidden1; ++i ) dense1_[j][i] = 0;
        }
        for ( int a = 0; a < kActions; ++a )
        {
            bPolicy_[a] = 0;
            for ( int i = 0; i < kHidden2; ++i ) policy_[a][i] = 0;
        }
        for ( int i = 0; i < kHidden2; ++i ) value_[i] = 0;
        bValue_ = 0;
    }

    WeightsView CurrentView() const
    {
        WeightsView view;
        view.conv1 = conv1_;
        view.bConv1 = bConv1_;
        view.conv2 = conv2_;
        view.bConv2 = bConv2_;
        view.dense0 = dense0_;
        view.bDense0 = bDense0_;
        view.dense1 = dense1_;
        view.bDense1 = bDense1_;
        view.policy = policy_;
        view.bPolicy = bPolicy_;
        view.value = value_;
        view.bValue = bValue_;
        return view;
    }

    WeightsView GetView( int viewIndex ) const
    {
        if ( viewIndex >= 0 && viewIndex < static_cast< int >( snapshots_.size() ) )
        {
            Snapshot const & snapshot = snapshots_[viewIndex];
            WeightsView view;
            view.conv1 = snapshot.conv1;
            view.bConv1 = snapshot.bConv1;
            view.conv2 = snapshot.conv2;
            view.bConv2 = snapshot.bConv2;
            view.dense0 = snapshot.dense0;
            view.bDense0 = snapshot.bDense0;
            view.dense1 = snapshot.dense1;
            view.bDense1 = snapshot.bDense1;
            view.policy = snapshot.policy;
            view.bPolicy = snapshot.bPolicy;
            view.value = snapshot.value;
            view.bValue = snapshot.bValue;
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

        for ( int oc = 0; oc < kConv1Channels; ++oc )
        {
            snapshot.bConv1[oc] = bConv1_[oc];
            for ( int ic = 0; ic < kTeacherMapChannels; ++ic )
                for ( int ky = 0; ky < kConvKernelSize; ++ky )
                    for ( int kx = 0; kx < kConvKernelSize; ++kx )
                        snapshot.conv1[oc][ic][ky][kx] = conv1_[oc][ic][ky][kx];
        }

        for ( int oc = 0; oc < kConv2Channels; ++oc )
        {
            snapshot.bConv2[oc] = bConv2_[oc];
            for ( int ic = 0; ic < kConv1Channels; ++ic )
                for ( int ky = 0; ky < kConvKernelSize; ++ky )
                    for ( int kx = 0; kx < kConvKernelSize; ++kx )
                        snapshot.conv2[oc][ic][ky][kx] = conv2_[oc][ic][ky][kx];
        }

        for ( int j = 0; j < kHidden1; ++j )
        {
            snapshot.bDense0[j] = bDense0_[j];
            for ( int i = 0; i < kDenseInput; ++i ) snapshot.dense0[j][i] = dense0_[j][i];
        }
        for ( int j = 0; j < kHidden2; ++j )
        {
            snapshot.bDense1[j] = bDense1_[j];
            for ( int i = 0; i < kHidden1; ++i ) snapshot.dense1[j][i] = dense1_[j][i];
        }
        for ( int a = 0; a < kActions; ++a )
        {
            snapshot.bPolicy[a] = bPolicy_[a];
            for ( int i = 0; i < kHidden2; ++i ) snapshot.policy[a][i] = policy_[a][i];
        }
        snapshot.bValue = bValue_;
        for ( int i = 0; i < kHidden2; ++i ) snapshot.value[i] = value_[i];
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

    void Forward( WeightsView const & view, REAL const x[kFeatures], bool canLeft, bool canRight, ForwardCache & cache ) const
    {
        UnpackTeacherInput( x, cache.scalars, cache.mapIn );

        for ( int oc = 0; oc < kConv1Channels; ++oc )
        {
            for ( int y = 0; y < kTeacherMapSize; ++y )
            {
                for ( int cellX = 0; cellX < kTeacherMapSize; ++cellX )
                {
                    REAL sum = view.bConv1[oc];
                    for ( int ic = 0; ic < kTeacherMapChannels; ++ic )
                    {
                        for ( int ky = 0; ky < kConvKernelSize; ++ky )
                        {
                            int inputY = y + ky - 1;
                            if ( inputY < 0 || inputY >= kTeacherMapSize ) continue;
                            for ( int kx = 0; kx < kConvKernelSize; ++kx )
                            {
                                int inputX = cellX + kx - 1;
                                if ( inputX < 0 || inputX >= kTeacherMapSize ) continue;
                                sum += view.conv1[oc][ic][ky][kx] * cache.mapIn[ic][inputY][inputX];
                            }
                        }
                    }
                    cache.conv1[oc][y][cellX] = std::tanh( sum );
                }
            }
        }

        for ( int oc = 0; oc < kConv2Channels; ++oc )
        {
            for ( int y = 0; y < kTeacherMapSize; ++y )
            {
                for ( int cellX = 0; cellX < kTeacherMapSize; ++cellX )
                {
                    REAL sum = view.bConv2[oc];
                    for ( int ic = 0; ic < kConv1Channels; ++ic )
                    {
                        for ( int ky = 0; ky < kConvKernelSize; ++ky )
                        {
                            int inputY = y + ky - 1;
                            if ( inputY < 0 || inputY >= kTeacherMapSize ) continue;
                            for ( int kx = 0; kx < kConvKernelSize; ++kx )
                            {
                                int inputX = cellX + kx - 1;
                                if ( inputX < 0 || inputX >= kTeacherMapSize ) continue;
                                sum += view.conv2[oc][ic][ky][kx] * cache.conv1[ic][inputY][inputX];
                            }
                        }
                    }
                    cache.conv2[oc][y][cellX] = std::tanh( sum );
                }
            }
        }

        int denseIndex = 0;
        for ( int i = 0; i < kTeacherScalarFeatures; ++i )
        {
            cache.denseIn[denseIndex++] = cache.scalars[i];
        }
        for ( int oc = 0; oc < kConv2Channels; ++oc )
            for ( int y = 0; y < kTeacherMapSize; ++y )
                for ( int cellX = 0; cellX < kTeacherMapSize; ++cellX )
                    cache.denseIn[denseIndex++] = cache.conv2[oc][y][cellX];

        for ( int j = 0; j < kHidden1; ++j )
        {
            REAL sum = view.bDense0[j];
            for ( int i = 0; i < kDenseInput; ++i ) sum += view.dense0[j][i] * cache.denseIn[i];
            cache.h0[j] = std::tanh( sum );
        }

        for ( int j = 0; j < kHidden2; ++j )
        {
            REAL sum = view.bDense1[j];
            for ( int i = 0; i < kHidden1; ++i ) sum += view.dense1[j][i] * cache.h0[i];
            cache.h1[j] = std::tanh( sum );
        }

        REAL logits[kActions];
        for ( int a = 0; a < kActions; ++a )
        {
            REAL sum = view.bPolicy[a];
            for ( int i = 0; i < kHidden2; ++i ) sum += view.policy[a][i] * cache.h1[i];
            logits[a] = sum;
        }
        if ( !canLeft ) logits[0] = -1E+20f;
        if ( !canRight ) logits[2] = -1E+20f;

        REAL maxLogit = logits[0];
        for ( int a = 1; a < kActions; ++a ) if ( logits[a] > maxLogit ) maxLogit = logits[a];

        REAL sumExp = 0;
        for ( int a = 0; a < kActions; ++a )
        {
            cache.p[a] = logits[a] < -1E+10f ? 0.0f : std::exp( logits[a] - maxLogit );
            sumExp += cache.p[a];
        }
        if ( sumExp <= 0 )
        {
            cache.p[0] = canLeft ? 0.5f : 0.0f;
            cache.p[1] = 1.0f;
            cache.p[2] = canRight ? 0.5f : 0.0f;
            sumExp = cache.p[0] + cache.p[1] + cache.p[2];
        }
        for ( int a = 0; a < kActions; ++a ) cache.p[a] /= sumExp;

        REAL value = view.bValue;
        for ( int i = 0; i < kHidden2; ++i ) value += view.value[i] * cache.h1[i];
        cache.v = value;
    }

    void Backward( ForwardCache const & cache, int action, REAL policyScale, REAL valueScale )
    {
        REAL dPolicy[kActions];
        for ( int a = 0; a < kActions; ++a )
        {
            REAL target = ( a == action ) ? 1.0f : 0.0f;
            dPolicy[a] = ( target - cache.p[a] ) * policyScale;
        }

        REAL dHidden1[kHidden2];
        for ( int j = 0; j < kHidden2; ++j )
        {
            REAL back = valueScale * value_[j];
            for ( int a = 0; a < kActions; ++a ) back += dPolicy[a] * policy_[a][j];
            dHidden1[j] = ActivationDerivative( cache.h1[j] ) * back;
        }

        REAL dHidden0[kHidden1];
        for ( int j = 0; j < kHidden1; ++j )
        {
            REAL back = 0;
            for ( int k = 0; k < kHidden2; ++k ) back += dHidden1[k] * dense1_[k][j];
            dHidden0[j] = ActivationDerivative( cache.h0[j] ) * back;
        }

        REAL dDenseIn[kDenseInput];
        for ( int i = 0; i < kDenseInput; ++i )
        {
            REAL back = 0;
            for ( int j = 0; j < kHidden1; ++j ) back += dHidden0[j] * dense0_[j][i];
            dDenseIn[i] = back;
        }

        REAL dConv2[kConv2Channels][kTeacherMapSize][kTeacherMapSize];
        for ( int oc = 0; oc < kConv2Channels; ++oc )
            for ( int y = 0; y < kTeacherMapSize; ++y )
                for ( int cellX = 0; cellX < kTeacherMapSize; ++cellX )
                    dConv2[oc][y][cellX] = 0;

        int denseIndex = kTeacherScalarFeatures;
        for ( int oc = 0; oc < kConv2Channels; ++oc )
            for ( int y = 0; y < kTeacherMapSize; ++y )
                for ( int cellX = 0; cellX < kTeacherMapSize; ++cellX )
                    dConv2[oc][y][cellX] = dDenseIn[denseIndex++];

        REAL dConv2Raw[kConv2Channels][kTeacherMapSize][kTeacherMapSize];
        for ( int oc = 0; oc < kConv2Channels; ++oc )
            for ( int y = 0; y < kTeacherMapSize; ++y )
                for ( int cellX = 0; cellX < kTeacherMapSize; ++cellX )
                    dConv2Raw[oc][y][cellX] = ActivationDerivative( cache.conv2[oc][y][cellX] ) * dConv2[oc][y][cellX];

        REAL dConv1[kConv1Channels][kTeacherMapSize][kTeacherMapSize];
        for ( int oc = 0; oc < kConv1Channels; ++oc )
            for ( int y = 0; y < kTeacherMapSize; ++y )
                for ( int cellX = 0; cellX < kTeacherMapSize; ++cellX )
                    dConv1[oc][y][cellX] = 0;

        for ( int oc = 0; oc < kConv2Channels; ++oc )
        {
            for ( int y = 0; y < kTeacherMapSize; ++y )
            {
                for ( int cellX = 0; cellX < kTeacherMapSize; ++cellX )
                {
                    REAL grad = dConv2Raw[oc][y][cellX];
                    for ( int ic = 0; ic < kConv1Channels; ++ic )
                    {
                        for ( int ky = 0; ky < kConvKernelSize; ++ky )
                        {
                            int inputY = y + ky - 1;
                            if ( inputY < 0 || inputY >= kTeacherMapSize ) continue;
                            for ( int kx = 0; kx < kConvKernelSize; ++kx )
                            {
                                int inputX = cellX + kx - 1;
                                if ( inputX < 0 || inputX >= kTeacherMapSize ) continue;
                                dConv1[ic][inputY][inputX] += grad * conv2_[oc][ic][ky][kx];
                            }
                        }
                    }
                }
            }
        }

        REAL dConv1Raw[kConv1Channels][kTeacherMapSize][kTeacherMapSize];
        for ( int oc = 0; oc < kConv1Channels; ++oc )
            for ( int y = 0; y < kTeacherMapSize; ++y )
                for ( int cellX = 0; cellX < kTeacherMapSize; ++cellX )
                    dConv1Raw[oc][y][cellX] = ActivationDerivative( cache.conv1[oc][y][cellX] ) * dConv1[oc][y][cellX];

        for ( int a = 0; a < kActions; ++a )
        {
            bPolicy_[a] += dPolicy[a];
            for ( int i = 0; i < kHidden2; ++i ) policy_[a][i] += dPolicy[a] * cache.h1[i];
        }

        bValue_ += valueScale;
        for ( int i = 0; i < kHidden2; ++i ) value_[i] += valueScale * cache.h1[i];

        for ( int j = 0; j < kHidden2; ++j )
        {
            bDense1_[j] += dHidden1[j];
            for ( int i = 0; i < kHidden1; ++i ) dense1_[j][i] += dHidden1[j] * cache.h0[i];
        }

        for ( int j = 0; j < kHidden1; ++j )
        {
            bDense0_[j] += dHidden0[j];
            for ( int i = 0; i < kDenseInput; ++i ) dense0_[j][i] += dHidden0[j] * cache.denseIn[i];
        }

        for ( int oc = 0; oc < kConv2Channels; ++oc )
        {
            for ( int y = 0; y < kTeacherMapSize; ++y )
            {
                for ( int cellX = 0; cellX < kTeacherMapSize; ++cellX )
                {
                    REAL grad = dConv2Raw[oc][y][cellX];
                    bConv2_[oc] += grad;
                    for ( int ic = 0; ic < kConv1Channels; ++ic )
                    {
                        for ( int ky = 0; ky < kConvKernelSize; ++ky )
                        {
                            int inputY = y + ky - 1;
                            if ( inputY < 0 || inputY >= kTeacherMapSize ) continue;
                            for ( int kx = 0; kx < kConvKernelSize; ++kx )
                            {
                                int inputX = cellX + kx - 1;
                                if ( inputX < 0 || inputX >= kTeacherMapSize ) continue;
                                conv2_[oc][ic][ky][kx] += grad * cache.conv1[ic][inputY][inputX];
                            }
                        }
                    }
                }
            }
        }

        for ( int oc = 0; oc < kConv1Channels; ++oc )
        {
            for ( int y = 0; y < kTeacherMapSize; ++y )
            {
                for ( int cellX = 0; cellX < kTeacherMapSize; ++cellX )
                {
                    REAL grad = dConv1Raw[oc][y][cellX];
                    bConv1_[oc] += grad;
                    for ( int ic = 0; ic < kTeacherMapChannels; ++ic )
                    {
                        for ( int ky = 0; ky < kConvKernelSize; ++ky )
                        {
                            int inputY = y + ky - 1;
                            if ( inputY < 0 || inputY >= kTeacherMapSize ) continue;
                            for ( int kx = 0; kx < kConvKernelSize; ++kx )
                            {
                                int inputX = cellX + kx - 1;
                                if ( inputX < 0 || inputX >= kTeacherMapSize ) continue;
                                conv1_[oc][ic][ky][kx] += grad * cache.mapIn[ic][inputY][inputX];
                            }
                        }
                    }
                }
            }
        }
    }

    void ApplyWeightDecay( REAL lr )
    {
        REAL shrink = 1.0f - lr * Clamp( sg_weightDecay, 0.0f, 1.0f );

        for ( int oc = 0; oc < kConv1Channels; ++oc )
            for ( int ic = 0; ic < kTeacherMapChannels; ++ic )
                for ( int ky = 0; ky < kConvKernelSize; ++ky )
                    for ( int kx = 0; kx < kConvKernelSize; ++kx )
                        conv1_[oc][ic][ky][kx] = Clamp( conv1_[oc][ic][ky][kx] * shrink, -sg_weightClip, sg_weightClip );

        for ( int oc = 0; oc < kConv2Channels; ++oc )
            for ( int ic = 0; ic < kConv1Channels; ++ic )
                for ( int ky = 0; ky < kConvKernelSize; ++ky )
                    for ( int kx = 0; kx < kConvKernelSize; ++kx )
                        conv2_[oc][ic][ky][kx] = Clamp( conv2_[oc][ic][ky][kx] * shrink, -sg_weightClip, sg_weightClip );

        for ( int j = 0; j < kHidden1; ++j )
            for ( int i = 0; i < kDenseInput; ++i )
                dense0_[j][i] = Clamp( dense0_[j][i] * shrink, -sg_weightClip, sg_weightClip );

        for ( int j = 0; j < kHidden2; ++j )
            for ( int i = 0; i < kHidden1; ++i )
                dense1_[j][i] = Clamp( dense1_[j][i] * shrink, -sg_weightClip, sg_weightClip );

        for ( int a = 0; a < kActions; ++a )
            for ( int i = 0; i < kHidden2; ++i )
                policy_[a][i] = Clamp( policy_[a][i] * shrink, -sg_weightClip, sg_weightClip );

        for ( int i = 0; i < kHidden2; ++i )
            value_[i] = Clamp( value_[i] * shrink, -sg_weightClip, sg_weightClip );
    }

    bool Load()
    {
        std::ifstream in;
        if ( !tDirectories::Var().Open( in, static_cast< char const * >( sg_modelFile ) ) ) return false;

        std::string magic;
        int scalarCount = 0, mapSize = 0, mapChannels = 0, conv1Channels = 0, conv2Channels = 0, h0 = 0, h1 = 0, actions = 0;
        in >> magic >> scalarCount >> mapSize >> mapChannels >> conv1Channels >> conv2Channels >> h0 >> h1 >> actions;
        if ( magic != "ARMAGETRON_TRAINED_AI_CNN_V1" ||
             scalarCount != kTeacherScalarFeatures || mapSize != kTeacherMapSize || mapChannels != kTeacherMapChannels ||
             conv1Channels != kConv1Channels || conv2Channels != kConv2Channels ||
             h0 != kHidden1 || h1 != kHidden2 || actions != kActions )
        {
            return false;
        }
        in >> baseline_ >> episodes_ >> updates_;
        if ( in.fail() ) return false;

        for ( int oc = 0; oc < kConv1Channels; ++oc )
        {
            for ( int ic = 0; ic < kTeacherMapChannels; ++ic )
                for ( int ky = 0; ky < kConvKernelSize; ++ky )
                    for ( int kx = 0; kx < kConvKernelSize; ++kx )
                        in >> conv1_[oc][ic][ky][kx];
            in >> bConv1_[oc];
        }

        for ( int oc = 0; oc < kConv2Channels; ++oc )
        {
            for ( int ic = 0; ic < kConv1Channels; ++ic )
                for ( int ky = 0; ky < kConvKernelSize; ++ky )
                    for ( int kx = 0; kx < kConvKernelSize; ++kx )
                        in >> conv2_[oc][ic][ky][kx];
            in >> bConv2_[oc];
        }

        for ( int j = 0; j < kHidden1; ++j )
        {
            for ( int i = 0; i < kDenseInput; ++i ) in >> dense0_[j][i];
            in >> bDense0_[j];
        }
        for ( int j = 0; j < kHidden2; ++j )
        {
            for ( int i = 0; i < kHidden1; ++i ) in >> dense1_[j][i];
            in >> bDense1_[j];
        }
        for ( int a = 0; a < kActions; ++a )
        {
            for ( int i = 0; i < kHidden2; ++i ) in >> policy_[a][i];
            in >> bPolicy_[a];
        }
        for ( int i = 0; i < kHidden2; ++i ) in >> value_[i];
        in >> bValue_;
        return !in.fail();
    }

    bool SaveToFile( char const * fileName ) const
    {
        std::ofstream out;
        if ( !tDirectories::Var().Open( out, fileName, std::ios::trunc ) ) return false;

        out.setf( std::ios::fixed );
        out.precision( 9 );
        out << "ARMAGETRON_TRAINED_AI_CNN_V1\n";
        out << kTeacherScalarFeatures << " " << kTeacherMapSize << " " << kTeacherMapChannels << " "
            << kConv1Channels << " " << kConv2Channels << " "
            << kHidden1 << " " << kHidden2 << " " << kActions << "\n";
        out << baseline_ << " " << episodes_ << " " << updates_ << "\n";

        for ( int oc = 0; oc < kConv1Channels; ++oc )
        {
            for ( int ic = 0; ic < kTeacherMapChannels; ++ic )
                for ( int ky = 0; ky < kConvKernelSize; ++ky )
                    for ( int kx = 0; kx < kConvKernelSize; ++kx )
                        out << conv1_[oc][ic][ky][kx] << " ";
            out << bConv1_[oc] << "\n";
        }
        for ( int oc = 0; oc < kConv2Channels; ++oc )
        {
            for ( int ic = 0; ic < kConv1Channels; ++ic )
                for ( int ky = 0; ky < kConvKernelSize; ++ky )
                    for ( int kx = 0; kx < kConvKernelSize; ++kx )
                        out << conv2_[oc][ic][ky][kx] << " ";
            out << bConv2_[oc] << "\n";
        }
        for ( int j = 0; j < kHidden1; ++j )
        {
            for ( int i = 0; i < kDenseInput; ++i ) out << dense0_[j][i] << " ";
            out << bDense0_[j] << "\n";
        }
        for ( int j = 0; j < kHidden2; ++j )
        {
            for ( int i = 0; i < kHidden1; ++i ) out << dense1_[j][i] << " ";
            out << bDense1_[j] << "\n";
        }
        for ( int a = 0; a < kActions; ++a )
        {
            for ( int i = 0; i < kHidden2; ++i ) out << policy_[a][i] << " ";
            out << bPolicy_[a] << "\n";
        }
        for ( int i = 0; i < kHidden2; ++i ) out << value_[i] << " ";
        out << bValue_ << "\n";
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
        for ( int oc = 0; oc < kConv1Channels; ++oc )
        {
            bConv1_[oc] = 0;
            for ( int ic = 0; ic < kTeacherMapChannels; ++ic )
                for ( int ky = 0; ky < kConvKernelSize; ++ky )
                    for ( int kx = 0; kx < kConvKernelSize; ++kx )
                        conv1_[oc][ic][ky][kx] = RandomSigned() * 0.05f;
        }
        for ( int oc = 0; oc < kConv2Channels; ++oc )
        {
            bConv2_[oc] = 0;
            for ( int ic = 0; ic < kConv1Channels; ++ic )
                for ( int ky = 0; ky < kConvKernelSize; ++ky )
                    for ( int kx = 0; kx < kConvKernelSize; ++kx )
                        conv2_[oc][ic][ky][kx] = RandomSigned() * 0.05f;
        }
        for ( int j = 0; j < kHidden1; ++j )
        {
            bDense0_[j] = RandomSigned() * 0.05f;
            for ( int i = 0; i < kDenseInput; ++i ) dense0_[j][i] = RandomSigned() * 0.05f;
        }
        for ( int j = 0; j < kHidden2; ++j )
        {
            bDense1_[j] = RandomSigned() * 0.05f;
            for ( int i = 0; i < kHidden1; ++i ) dense1_[j][i] = RandomSigned() * 0.05f;
        }
        for ( int a = 0; a < kActions; ++a )
        {
            bPolicy_[a] = 0;
            for ( int i = 0; i < kHidden2; ++i ) policy_[a][i] = RandomSigned() * 0.05f;
        }
        bValue_ = 0;
        for ( int i = 0; i < kHidden2; ++i ) value_[i] = RandomSigned() * 0.05f;
        bPolicy_[1] = 0.2f;
    }

    REAL conv1_[kConv1Channels][kTeacherMapChannels][kConvKernelSize][kConvKernelSize];
    REAL bConv1_[kConv1Channels];
    REAL conv2_[kConv2Channels][kConv1Channels][kConvKernelSize][kConvKernelSize];
    REAL bConv2_[kConv2Channels];
    REAL dense0_[kHidden1][kDenseInput];
    REAL bDense0_[kHidden1];
    REAL dense1_[kHidden2][kHidden1];
    REAL bDense1_[kHidden2];
    REAL policy_[kActions][kHidden2];
    REAL bPolicy_[kActions];
    REAL value_[kHidden2];
    REAL bValue_;
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
        : teacher( false ),
          survived( false ),
          distance( 0 ),
          reward( 0 )
    {
    }

    bool teacher;
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
          learnedEpisodes_( 0 ),
          rewardTotal_( 0 ),
          distanceTotal_( 0 ),
          predictedValueTotal_( 0 ),
          stepsTotal_( 0 ),
          policyLossTotal_( 0 ),
          valueLossTotal_( 0 ),
          entropyTotal_( 0 ),
          lastSurvived_( false ),
          lastReward_( 0 ),
          lastDistance_( 0 ),
          lastAveragePredictedValue_( 0 ),
          lastSteps_( 0 ),
          lastLearned_( false ),
          lastPolicyLoss_( 0 ),
          lastValueLoss_( 0 ),
          lastEntropy_( 0 )
    {
    }

    void EnsureSource( std::string const & path )
    {
        if ( offsets_.find( path ) == offsets_.end() )
        {
            offsets_[path] = 0;
        }
        if ( sourceEpisodes_.find( path ) == sourceEpisodes_.end() )
        {
            sourceEpisodes_[path] = 0;
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

    void AddSourceEpisodes( std::string const & path, unsigned int episodes )
    {
        sourceEpisodes_[path] += episodes;
    }

    void RecordEpisode( bool survived, REAL distance, REAL reward, REAL averagePredictedValue, unsigned int steps, LearningStats const & learningStats, bool learned )
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
        if ( learningStats.valid )
        {
            ++learnedEpisodes_;
            policyLossTotal_ += learningStats.policyLoss;
            valueLossTotal_ += learningStats.valueLoss;
            entropyTotal_ += learningStats.entropy;
            lastPolicyLoss_ = learningStats.policyLoss;
            lastValueLoss_ = learningStats.valueLoss;
            lastEntropy_ = learningStats.entropy;
        }
        lastSurvived_ = survived;
        lastReward_ = reward;
        lastDistance_ = distance;
        lastAveragePredictedValue_ = averagePredictedValue;
        lastSteps_ = steps;
        lastLearned_ = learned;
    }

    unsigned long long MetricsEpisodes() const { return metricsEpisodes_; }
    unsigned long long Wins() const { return wins_; }
    unsigned long long LearnedEpisodes() const { return learnedEpisodes_; }
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
    REAL LastPolicyLoss() const { return lastPolicyLoss_; }
    REAL LastValueLoss() const { return lastValueLoss_; }
    REAL LastEntropy() const { return lastEntropy_; }

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

    REAL AveragePolicyLoss() const
    {
        if ( learnedEpisodes_ <= 0 )
        {
            return 0;
        }

        return policyLossTotal_ / static_cast< REAL >( learnedEpisodes_ );
    }

    REAL AverageValueLoss() const
    {
        if ( learnedEpisodes_ <= 0 )
        {
            return 0;
        }

        return valueLossTotal_ / static_cast< REAL >( learnedEpisodes_ );
    }

    REAL AverageEntropy() const
    {
        if ( learnedEpisodes_ <= 0 )
        {
            return 0;
        }

        return entropyTotal_ / static_cast< REAL >( learnedEpisodes_ );
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
            else if ( key == "learned_episodes" )
            {
                in >> learnedEpisodes_;
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
            else if ( key == "policy_loss_total" )
            {
                in >> policyLossTotal_;
            }
            else if ( key == "value_loss_total" )
            {
                in >> valueLossTotal_;
            }
            else if ( key == "entropy_total" )
            {
                in >> entropyTotal_;
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
            else if ( key == "last_policy_loss" )
            {
                in >> lastPolicyLoss_;
            }
            else if ( key == "last_value_loss" )
            {
                in >> lastValueLoss_;
            }
            else if ( key == "last_entropy" )
            {
                in >> lastEntropy_;
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
            else if ( key == "source_episodes" )
            {
                std::string path;
                unsigned long long episodes = 0;
                in >> path >> episodes;
                sourceEpisodes_[path] = episodes;
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
        out << "learned_episodes " << learnedEpisodes_ << "\n";
        out << "reward_total " << rewardTotal_ << "\n";
        out << "distance_total " << distanceTotal_ << "\n";
        out << "predicted_value_total " << predictedValueTotal_ << "\n";
        out << "steps_total " << stepsTotal_ << "\n";
        out << "policy_loss_total " << policyLossTotal_ << "\n";
        out << "value_loss_total " << valueLossTotal_ << "\n";
        out << "entropy_total " << entropyTotal_ << "\n";
        out << "last_survived " << ( lastSurvived_ ? 1 : 0 ) << "\n";
        out << "last_reward " << lastReward_ << "\n";
        out << "last_distance " << lastDistance_ << "\n";
        out << "last_average_predicted_value " << lastAveragePredictedValue_ << "\n";
        out << "last_steps " << lastSteps_ << "\n";
        out << "last_policy_loss " << lastPolicyLoss_ << "\n";
        out << "last_value_loss " << lastValueLoss_ << "\n";
        out << "last_entropy " << lastEntropy_ << "\n";
        out << "last_learned " << ( lastLearned_ ? 1 : 0 ) << "\n";

        for ( std::map< std::string, long long >::const_iterator it = offsets_.begin(); it != offsets_.end(); ++it )
        {
            out << "source " << it->first << " " << it->second << "\n";
        }
        for ( std::map< std::string, unsigned long long >::const_iterator it = sourceEpisodes_.begin(); it != sourceEpisodes_.end(); ++it )
        {
            out << "source_episodes " << it->first << " " << it->second << "\n";
        }
    }

private:
    unsigned long long metricsEpisodes_;
    unsigned long long wins_;
    unsigned long long learnedEpisodes_;
    REAL rewardTotal_;
    REAL distanceTotal_;
    REAL predictedValueTotal_;
    unsigned long long stepsTotal_;
    REAL policyLossTotal_;
    REAL valueLossTotal_;
    REAL entropyTotal_;
    bool lastSurvived_;
    REAL lastReward_;
    REAL lastDistance_;
    REAL lastAveragePredictedValue_;
    unsigned int lastSteps_;
    bool lastLearned_;
    REAL lastPolicyLoss_;
    REAL lastValueLoss_;
    REAL lastEntropy_;
    std::map< std::string, long long > offsets_;
    std::map< std::string, unsigned long long > sourceEpisodes_;
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

static void AppendOfflineMetricsCsv( OfflineTrainerState const & state, bool survived, REAL distance, REAL reward, REAL averagePredictedValue, unsigned int steps, LearningStats const & learningStats, bool learned, unsigned int policyEpisodes, unsigned int policyUpdates )
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
        out << "episode,survived,reward,distance,average_predicted_value,steps,learned,policy_episodes,policy_updates,cumulative_win_rate,cumulative_average_reward,cumulative_average_distance,cumulative_average_predicted_value,policy_loss,value_loss,entropy,cumulative_average_policy_loss,cumulative_average_value_loss,cumulative_average_entropy\n";
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
        << "," << ( learningStats.valid ? learningStats.policyLoss : 0 )
        << "," << ( learningStats.valid ? learningStats.valueLoss : 0 )
        << "," << ( learningStats.valid ? learningStats.entropy : 0 )
        << "," << state.AveragePolicyLoss()
        << "," << state.AverageValueLoss()
        << "," << state.AverageEntropy()
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
    out << "steps_total " << state.StepsTotal() << "\n";
    out << "learned_episodes " << state.LearnedEpisodes() << "\n";
    out << "average_policy_loss " << state.AveragePolicyLoss() << "\n";
    out << "average_value_loss " << state.AverageValueLoss() << "\n";
    out << "average_entropy " << state.AverageEntropy() << "\n";
    out << "last_survived " << ( state.LastSurvived() ? 1 : 0 ) << "\n";
    out << "last_reward " << state.LastReward() << "\n";
    out << "last_distance " << state.LastDistance() << "\n";
    out << "last_average_predicted_value " << state.LastAveragePredictedValue() << "\n";
    out << "last_steps " << state.LastSteps() << "\n";
    out << "last_policy_loss " << state.LastPolicyLoss() << "\n";
    out << "last_value_loss " << state.LastValueLoss() << "\n";
    out << "last_entropy " << state.LastEntropy() << "\n";
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

static bool ParseTeacherRecordedStep( std::string const & line, unsigned long long & episodeId, Step & step )
{
    std::istringstream in( line );
    std::string tag;
    unsigned int stepIndex = 0;
    int action = 1;
    int canLeft = 0;
    int canRight = 0;
    REAL halfExtent = 0;

    if ( !( in >> tag >> episodeId >> stepIndex >> action >> canLeft >> canRight >> halfExtent ) )
    {
        return false;
    }
    if ( tag != "teacher_v1" )
    {
        return false;
    }

    step.action = action;
    step.canLeft = canLeft != 0;
    step.canRight = canRight != 0;
    step.v = 0;
    for ( int a = 0; a < kActions; ++a )
    {
        step.p[a] = 0;
    }

    int index = 0;
    for ( int i = 0; i < kTeacherScalarFeatures; ++i )
    {
        if ( !( in >> step.x[index++] ) )
        {
            return false;
        }
    }

    for ( int channel = 0; channel < kTeacherMapChannels; ++channel )
        for ( int y = 0; y < kTeacherMapSize; ++y )
            for ( int cellX = 0; cellX < kTeacherMapSize; ++cellX )
                if ( !( in >> step.x[index++] ) )
                {
                    return false;
                }

    return index == kFeatures;
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

    REAL averagePredictedValue = episodeData.teacher ? 0 : AveragePredictedValue( episodeData.steps );
    LearningStats learningStats = episodeData.teacher
        ? Policy::Get().TrainTeacher( episodeData.steps )
        : Policy::Get().Train( episodeData.steps, episodeData.reward );
    state.RecordEpisode(
        episodeData.survived,
        episodeData.distance,
        episodeData.reward,
        averagePredictedValue,
        static_cast< unsigned int >( episodeData.steps.size() ),
        learningStats,
        true );
    AppendOfflineMetricsCsv(
        state,
        episodeData.survived,
        episodeData.distance,
        episodeData.reward,
        averagePredictedValue,
        static_cast< unsigned int >( episodeData.steps.size() ),
        learningStats,
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
        bool isTeacher = false;
        if ( line.compare( 0, 10, "teacher_v1" ) == 0 )
        {
            isTeacher = true;
            if ( !ParseTeacherRecordedStep( line, episodeId, step ) )
            {
                continue;
            }
        }
        else if ( !ParseOfflineRecordedStep( line, episodeId, survived, distance, reward, step ) )
        {
            continue;
        }

        if ( !hasEpisode )
        {
            hasEpisode = true;
            currentEpisodeId = episodeId;
            currentEpisode.teacher = isTeacher;
            currentEpisode.survived = survived;
            currentEpisode.distance = distance;
            currentEpisode.reward = reward;
            currentEpisode.steps.clear();
        }
        else if ( episodeId != currentEpisodeId || currentEpisode.teacher != isTeacher )
        {
            TrainOfflineEpisode( currentEpisode, state );
            ++trainedEpisodes;
            currentEpisodeId = episodeId;
            currentEpisode.teacher = isTeacher;
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

    state.AddSourceEpisodes( path, trainedEpisodes );
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
          trainThisEpisode_( Policy::Get().IsLiveView( policyView_ ) )
    {
        episode_.reserve( 512 );
    }

    virtual void OnRoundResult( bool survived, REAL distance ) override
    {
        if ( distance < 0 ) distance = 0;
        REAL reward = distance * sg_rewardDistance + ( survived ? sg_rewardWin : sg_rewardDeath );
        REAL averagePredictedValue = AveragePredictedValue( episode_ );
        RecordEpisode( episode_, survived, distance, reward );
        bool learned = false;
        LearningStats learningStats;
        if ( sg_learn && trainThisEpisode_ )
        {
            learningStats = Policy::Get().Train( episode_, reward );
            learned = true;
        }
        TrainingMetrics::Get().RecordEpisode(
            survived,
            distance,
            reward,
            averagePredictedValue,
            static_cast< unsigned int >( episode_.size() ),
            learningStats,
            learned,
            Policy::Get().Episodes(),
            Policy::Get().Updates() );
        episode_.clear();
        policyView_ = Policy::Get().AcquireView();
        trainThisEpisode_ = Policy::Get().IsLiveView( policyView_ );
    }

protected:
    virtual REAL DoThink() override
    {
        gCycle * cycle = Object();
        if ( !cycle || !cycle->Alive() ) return sg_thinkTime;

        TeacherExample example;
        BuildTeacherExample( cycle, 0, example );
        bool canLeft = example.canLeft;
        bool canRight = example.canRight;
        REAL x[kFeatures] = { 0 };
        FlattenTeacherExample( example, x );

        REAL p[kActions];
        REAL v = 0;
        int action = Policy::Get().Choose( policyView_, x, canLeft, canRight, p, v );
        int turn = ActionToTurn( action );
        if ( turn != 0 && cycle->CanMakeTurn( turn ) ) cycle->Turn( turn );

        if ( ( sg_learn || sg_record ) && static_cast< int >( episode_.size() ) < sg_maxEpisodeSteps )
        {
            Step s;
            for ( int i = 0; i < kFeatures; ++i ) s.x[i] = x[i];
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
    std::vector< Step > episode_;
    int policyView_;
    bool trainThisEpisode_;
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

static void RecordTeacherDecisionImpl( gCycle * cycle, int turn )
{
    if ( !sg_record || !cycle || !cycle->Alive() )
    {
        return;
    }

    char const * teacherFile = static_cast< char const * >( sg_teacherFile );
    if ( !teacherFile || !teacherFile[0] )
    {
        return;
    }

    TeacherEpisodeState & episode = GetTeacherEpisodeState( cycle );
    int stride = sg_recordStride < 1 ? 1 : sg_recordStride;
    if ( episode.stepIndex % static_cast< unsigned int >( stride ) != 0 )
    {
        ++episode.stepIndex;
        return;
    }

    bool canLeft = cycle->CanMakeTurn( -1 );
    bool canRight = cycle->CanMakeTurn( 1 );
    if ( turn < 0 && !canLeft )
    {
        turn = 0;
    }
    if ( turn > 0 && !canRight )
    {
        turn = 0;
    }

    TeacherExample example;
    BuildTeacherExample( cycle, turn, example );

    std::ofstream out;
    if ( !tDirectories::Var().Open( out, teacherFile, std::ios::app ) )
    {
        return;
    }

    out.setf( std::ios::fixed );
    out.precision( 6 );
    out << "teacher_v1"
        << " " << episode.episodeId
        << " " << episode.stepIndex
        << " " << example.action
        << " " << ( example.canLeft ? 1 : 0 )
        << " " << ( example.canRight ? 1 : 0 )
        << " " << example.halfExtent;
    for ( int i = 0; i < kTeacherScalarFeatures; ++i )
    {
        out << " " << example.scalars[i];
    }
    for ( int channel = 0; channel < kTeacherMapChannels; ++channel )
        for ( int y = 0; y < kTeacherMapSize; ++y )
            for ( int x = 0; x < kTeacherMapSize; ++x )
                out << " " << example.map[channel][y][x];
    out << "\n";

    ++episode.stepIndex;
}

static void RecordTeacherEpisodeResultImpl( gCycle * cycle, bool survived, REAL distance )
{
    (void)survived;
    (void)distance;
    if ( !cycle )
    {
        return;
    }

    sg_teacherEpisodes.erase( cycle );
}

static tConfItem< bool > sg_enableConf( "AI_TRAINED_ENABLE", sg_enable, &OnEnableChanged );
static tConfItem< bool > sg_learnConf( "AI_TRAINED_LEARN", sg_learn );
static tConfItem< bool > sg_recordConf( "AI_TRAINED_RECORD", sg_record );
static tConfItem< bool > sg_autostartConf( "AI_TRAINED_AUTOSTART", sg_autostart );
static tConfItem< int > sg_botCountConf( "AI_TRAINED_BOT_COUNT", sg_botCount );
static tConfItem< tString > sg_modelFileConf( "AI_TRAINED_MODEL_FILE", sg_modelFile );
static tConfItem< tString > sg_recordFileConf( "AI_TRAINED_RECORD_FILE", sg_recordFile );
static tConfItem< tString > sg_teacherFileConf( "AI_TRAINED_TEACHER_FILE", sg_teacherFile );
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

void gTrainedAI_RecordTeacherDecision( gCycle * cycle, int turn )
{
    RecordTeacherDecisionImpl( cycle, turn );
}

void gTrainedAI_RecordTeacherEpisodeResult( gCycle * cycle, bool survived, REAL distance )
{
    RecordTeacherEpisodeResultImpl( cycle, survived, distance );
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
