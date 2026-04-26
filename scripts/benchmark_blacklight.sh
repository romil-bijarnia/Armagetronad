#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/blacklight_lib.sh"
REPO_ROOT="${BLACKLIGHT_REPO_ROOT}"

BASE_CFG_REL="${ARMAGETRON_BENCH_BASE_CFG_REL:-examples/trained_ai_selfplay.cfg}"
BASE_CFG_PATH="${REPO_ROOT}/config/${BASE_CFG_REL}"
BENCH_ID="${ARMAGETRON_BENCH_ID:-$(date +%Y%m%d-%H%M%S)}"
BENCH_ROOT_REL="blacklight_bench/${BENCH_ID}"
BENCH_ROOT_ABS="${REPO_ROOT}/var/${BENCH_ROOT_REL}"

SUITE=""
CANDIDATE_SOURCE=""
REFERENCE_SOURCE=""
BIN_OVERRIDE="${ARMAGETRON_SELFPLAY_BIN:-}"
SESSIONS_OVERRIDE="${ARMAGETRON_BENCH_SESSIONS:-}"
DURATION_OVERRIDE="${ARMAGETRON_BENCH_DURATION_SECONDS:-}"
ROUNDS_OVERRIDE="${ARMAGETRON_BENCH_LIMIT_ROUNDS:-}"

usage() {
    cat <<'EOF'
Usage:
  ./scripts/benchmark_blacklight.sh --suite NAME --candidate MODEL_PATH_OR_RUN [--reference MODEL_PATH_OR_RUN] [--bin PATH]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite)
            SUITE="$2"
            shift 2
            ;;
        --candidate)
            CANDIDATE_SOURCE="$2"
            shift 2
            ;;
        --reference)
            REFERENCE_SOURCE="$2"
            shift 2
            ;;
        --bin)
            BIN_OVERRIDE="$2"
            shift 2
            ;;
        --sessions)
            SESSIONS_OVERRIDE="$2"
            shift 2
            ;;
        --duration)
            DURATION_OVERRIDE="$2"
            shift 2
            ;;
        --rounds)
            ROUNDS_OVERRIDE="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown benchmark option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

[[ -n "${SUITE}" ]] || { echo "Missing benchmark suite." >&2; exit 1; }
[[ -n "${CANDIDATE_SOURCE}" ]] || { echo "Missing benchmark candidate." >&2; exit 1; }

if ! blacklight_bench_suite_exists "${SUITE}"; then
    echo "Unknown benchmark suite: ${SUITE}" >&2
    exit 1
fi

if [[ ! -f "${BASE_CFG_PATH}" ]]; then
    echo "Missing benchmark base config: ${BASE_CFG_PATH}" >&2
    exit 1
fi

if ! CANDIDATE_MODEL="$(blacklight_resolve_model_input "${CANDIDATE_SOURCE}" 1)"; then
    echo "Could not resolve candidate model: ${CANDIDATE_SOURCE}" >&2
    exit 1
fi

if [[ -z "${REFERENCE_SOURCE}" ]]; then
    REFERENCE_SOURCE="$(blacklight_current_champion_model || true)"
fi

if [[ -n "${REFERENCE_SOURCE}" ]]; then
    if ! REFERENCE_MODEL="$(blacklight_resolve_model_input "${REFERENCE_SOURCE}" 1)"; then
        echo "Could not resolve reference model: ${REFERENCE_SOURCE}" >&2
        exit 1
    fi
else
    REFERENCE_MODEL=""
fi

if ! BIN_PATH="$(blacklight_find_server_bin "${BIN_OVERRIDE}")"; then
    echo "Could not find a dedicated-capable server binary for benchmark runs." >&2
    exit 1
fi
if ! blacklight_preflight_server_bin "${BIN_PATH}"; then
    exit 1
fi

SESSIONS="${SESSIONS_OVERRIDE:-$(blacklight_bench_suite_value "${SUITE}" sessions)}"
BENCH_DURATION_SECONDS="${DURATION_OVERRIDE:-$(blacklight_bench_suite_value "${SUITE}" duration_seconds)}"
BENCH_LIMIT_ROUNDS="${ROUNDS_OVERRIDE:-$(blacklight_bench_suite_value "${SUITE}" limit_rounds)}"
BOT_COUNT="$(blacklight_bench_suite_value "${SUITE}" bot_count)"
MIN_PLAYERS="$(blacklight_bench_suite_value "${SUITE}" min_players)"
TEAMS_MIN="$(blacklight_bench_suite_value "${SUITE}" teams_min)"
TEAMS_MAX="$(blacklight_bench_suite_value "${SUITE}" teams_max)"

if ! [[ "${SESSIONS}" =~ ^[0-9]+$ ]] || (( SESSIONS < 1 )); then
    echo "Benchmark sessions must be a positive integer." >&2
    exit 1
fi
if ! [[ "${BENCH_DURATION_SECONDS}" =~ ^[0-9]+$ ]] || (( BENCH_DURATION_SECONDS < 1 )); then
    echo "Benchmark duration must be a positive integer." >&2
    exit 1
fi
if ! [[ "${BENCH_LIMIT_ROUNDS}" =~ ^[0-9]+$ ]] || (( BENCH_LIMIT_ROUNDS < 1 )); then
    echo "Benchmark rounds must be a positive integer." >&2
    exit 1
fi

mkdir -p "${BENCH_ROOT_ABS}"

run_benchmark_session() {
    local label="$1"
    local session_index="$2"
    local model_abs="$3"
    local session_rel="${BENCH_ROOT_REL}/${label}/session$(printf '%02d' "${session_index}")"
    local session_abs="${REPO_ROOT}/var/${session_rel}"
    local cfg_rel="generated_bench_${BENCH_ID}_${label}_$(printf '%02d' "${session_index}").cfg"
    local cfg_path="${REPO_ROOT}/config/${cfg_rel}"
    local eval_metrics_rel="${session_rel}/ai_eval_metrics.csv"
    local eval_metrics_abs="${REPO_ROOT}/var/${eval_metrics_rel}"
    local eval_metrics_summary="${session_abs}/blacklight_eval_metrics.latest"
    local blacklight_metrics_rel="${session_rel}/blacklight_metrics.csv"
    local server_log="${session_abs}/server.log"
    local model_rel=""
    local deadline=0
    local blacklight_episodes=0

    mkdir -p "${session_abs}"
    model_rel="$(blacklight_copy_model_into_var "${model_abs}" "${session_rel}/model.txt")"

    cat > "${cfg_path}" <<EOF
SINCLUDE ${BASE_CFG_REL}
SERVER_NAME Blacklight Bench ${label} $(printf '%02d' "${session_index}")
AI_TRAINED_ENABLE 1
AI_TRAINED_MODEL_FILE ${model_rel}
AI_TRAINED_BOT_COUNT ${BOT_COUNT}
AI_TRAINED_LEARN 0
AI_TRAINED_RECORD 0
AI_TRAINED_METRICS_FILE ${blacklight_metrics_rel}
AI_TRAINED_POLICY_POOL_SIZE 0
AI_TRAINED_POLICY_HISTORIC_PROB 0
AI_EVAL_METRICS_FILE ${eval_metrics_rel}
WIN_ZONE_RANDOMNESS 0
LIMIT_ROUNDS ${BENCH_LIMIT_ROUNDS}
MIN_PLAYERS ${MIN_PLAYERS}
TEAMS_MIN ${TEAMS_MIN}
TEAMS_MAX ${TEAMS_MAX}
NUM_AIS 0
AUTO_AIS 0
EOF

    "${BIN_PATH}" \
        --datadir "${REPO_ROOT}" \
        --configdir "${REPO_ROOT}/config" \
        --userdatadir "${REPO_ROOT}/var" \
        --vardir "${REPO_ROOT}/var" \
        --extraconfig "${cfg_rel}" > "${server_log}" 2>&1 &
    local server_pid=$!
    deadline=$(( $(date +%s) + BENCH_DURATION_SECONDS ))

    while kill -0 "${server_pid}" 2>/dev/null; do
        if [[ -f "${eval_metrics_abs}" ]]; then
            blacklight_episodes="$(blacklight_eval_metrics_episodes "${eval_metrics_abs}" "BLACKLIGHT")"
            if [[ "${blacklight_episodes:-0}" =~ ^[0-9]+$ ]] && (( blacklight_episodes >= BENCH_LIMIT_ROUNDS )); then
                break
            fi
        fi

        if (( $(date +%s) >= deadline )); then
            break
        fi
        sleep 1
    done

    kill -TERM "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true

    if [[ ! -f "${eval_metrics_abs}" ]]; then
        echo "Benchmark session ${label}/$(printf '%02d' "${session_index}") did not produce ${eval_metrics_abs}" >&2
        return 1
    fi

    blacklight_write_eval_kind_summary "${eval_metrics_abs}" "BLACKLIGHT" "${eval_metrics_summary}"
    printf '%s\n' "${eval_metrics_summary}"
}

aggregate_summaries() {
    local prefix="$1"
    shift
    local count=0
    local total_episodes=0
    local win_values=""
    local distance_values=""
    local episodes
    local win_rate
    local distance
    local summary_path
    local mean_win
    local stddev_win
    local mean_distance
    local stddev_distance

    for summary_path in "$@"; do
        [[ -f "${summary_path}" ]] || continue
        episodes="$(blacklight_read_summary_value "${summary_path}" episodes)"
        win_rate="$(blacklight_read_summary_value "${summary_path}" win_rate)"
        distance="$(blacklight_read_summary_value "${summary_path}" average_distance)"

        [[ -n "${episodes}" ]] || episodes=0
        [[ -n "${win_rate}" ]] || win_rate=0
        [[ -n "${distance}" ]] || distance=0

        total_episodes=$(( total_episodes + episodes ))
        win_values="${win_values}${win_rate}"$'\n'
        distance_values="${distance_values}${distance}"$'\n'
        count=$(( count + 1 ))
    done

    if (( count == 0 )); then
        return 1
    fi

    mean_win="$(printf '%s' "${win_values}" | awk 'NF { sum += $1; count++ } END { if (count == 0) print "0.000000"; else printf "%.6f\n", sum / count }')"
    stddev_win="$(printf '%s' "${win_values}" | awk 'NF { sum += $1; sumsq += ($1 * $1); count++ } END { if (count == 0) print "0.000000"; else { mean = sum / count; variance = (sumsq / count) - (mean * mean); if (variance < 0) variance = 0; printf "%.6f\n", sqrt(variance) } }')"
    mean_distance="$(printf '%s' "${distance_values}" | awk 'NF { sum += $1; count++ } END { if (count == 0) print "0.000000"; else printf "%.6f\n", sum / count }')"
    stddev_distance="$(printf '%s' "${distance_values}" | awk 'NF { sum += $1; sumsq += ($1 * $1); count++ } END { if (count == 0) print "0.000000"; else { mean = sum / count; variance = (sumsq / count) - (mean * mean); if (variance < 0) variance = 0; printf "%.6f\n", sqrt(variance) } }')"

    cat <<EOF
${prefix}_sessions ${count}
${prefix}_total_episodes ${total_episodes}
${prefix}_mean_win_rate ${mean_win}
${prefix}_stddev_win_rate ${stddev_win}
${prefix}_mean_distance ${mean_distance}
${prefix}_stddev_distance ${stddev_distance}
EOF
}

declare -a CANDIDATE_SUMMARIES=()
declare -a REFERENCE_SUMMARIES=()

for (( session_index = 1; session_index <= SESSIONS; ++session_index )); do
    CANDIDATE_SUMMARIES+=( "$(run_benchmark_session candidate "${session_index}" "${CANDIDATE_MODEL}")" )
done

if [[ -n "${REFERENCE_MODEL}" ]]; then
    for (( session_index = 1; session_index <= SESSIONS; ++session_index )); do
        REFERENCE_SUMMARIES+=( "$(run_benchmark_session reference "${session_index}" "${REFERENCE_MODEL}")" )
    done
fi

REPORT_PATH="${BENCH_ROOT_ABS}/report.txt"
{
    echo "Blacklight benchmark"
    echo "suite ${SUITE}"
    echo "candidate_source ${CANDIDATE_SOURCE}"
    echo "candidate_model ${CANDIDATE_MODEL}"
    echo "report_path ${REPORT_PATH}"
    echo "bench_dir ${BENCH_ROOT_ABS}"
    echo "sessions ${SESSIONS}"
    echo "duration_seconds ${BENCH_DURATION_SECONDS}"
    echo "limit_rounds ${BENCH_LIMIT_ROUNDS}"
    echo "bot_count ${BOT_COUNT}"
    echo "min_players ${MIN_PLAYERS}"
    echo "teams_min ${TEAMS_MIN}"
    echo "teams_max ${TEAMS_MAX}"
    aggregate_summaries candidate "${CANDIDATE_SUMMARIES[@]}"

    if [[ -n "${REFERENCE_MODEL}" ]]; then
        echo "reference_source ${REFERENCE_SOURCE}"
        echo "reference_model ${REFERENCE_MODEL}"
        aggregate_summaries reference "${REFERENCE_SUMMARIES[@]}"
        awk '
            $1 == "candidate_mean_win_rate" { candidate_win = $2 }
            $1 == "reference_mean_win_rate" { reference_win = $2 }
            $1 == "candidate_mean_distance" { candidate_distance = $2 }
            $1 == "reference_mean_distance" { reference_distance = $2 }
            END {
                printf "delta_win_rate %.6f\n", candidate_win - reference_win
                if (reference_distance > 0) {
                    printf "distance_ratio %.6f\n", candidate_distance / reference_distance
                } else {
                    printf "distance_ratio 0.000000\n"
                }
            }
        ' <(aggregate_summaries candidate "${CANDIDATE_SUMMARIES[@]}") <(aggregate_summaries reference "${REFERENCE_SUMMARIES[@]}")
    fi
} > "${REPORT_PATH}"

cat "${REPORT_PATH}"
