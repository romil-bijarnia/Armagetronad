#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/blacklight_lib.sh"
REPO_ROOT="${BLACKLIGHT_REPO_ROOT}"

BASE_CFG_REL="${ARMAGETRON_SELFPLAY_BASE_CFG_REL:-examples/trained_ai_selfplay.cfg}"
BASE_CFG_PATH="${REPO_ROOT}/config/${BASE_CFG_REL}"
GENERATION="${ARMAGETRON_SELFPLAY_GENERATION:-blacklight_v10_parallel}"
RUN_ID="${ARMAGETRON_SELFPLAY_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${ARMAGETRON_SELFPLAY_RUN_NAME:-${GENERATION}_${RUN_ID}}"
RUN_DIR_REL="${ARMAGETRON_SELFPLAY_RUN_DIR_REL:-blacklight_runs/${RUN_NAME}}"
RUN_DIR_ABS="${REPO_ROOT}/var/${RUN_DIR_REL}"

MODEL_REL="${ARMAGETRON_SELFPLAY_MODEL_REL:-${RUN_DIR_REL}/trained_ai_model.txt}"
METRICS_REL="${ARMAGETRON_SELFPLAY_METRICS_REL:-${RUN_DIR_REL}/trained_ai_training_metrics.csv}"
CHECKPOINT_PREFIX_REL="${ARMAGETRON_SELFPLAY_CHECKPOINT_PREFIX_REL:-${RUN_DIR_REL}/checkpoints/blacklight}"
SOURCE_LIST_REL="${ARMAGETRON_SELFPLAY_OFFLINE_SOURCE_LIST_REL:-${RUN_DIR_REL}/offline_sources.txt}"
STATE_FILE_REL="${ARMAGETRON_SELFPLAY_OFFLINE_STATE_REL:-${RUN_DIR_REL}/trained_ai_offline_state.txt}"
PROGRESS_REL="${ARMAGETRON_SELFPLAY_PROGRESS_REL:-${RUN_DIR_REL}/training_progress.env}"
EVENTS_LOG_REL="${ARMAGETRON_SELFPLAY_EVENTS_LOG_REL:-${RUN_DIR_REL}/training_events.log}"
TRAINER_LOG_REL="${ARMAGETRON_SELFPLAY_TRAINER_LOG_REL:-${RUN_DIR_REL}/trainer_sync.log}"
CHECKPOINT_EVERY="${ARMAGETRON_SELFPLAY_CHECKPOINT_EVERY:-}"
SAVE_EVERY="${ARMAGETRON_SELFPLAY_SAVE_EVERY:-}"
LIMIT_ROUNDS="${ARMAGETRON_SELFPLAY_LIMIT_ROUNDS:-0}"
DURATION_SECONDS="${ARMAGETRON_SELFPLAY_DURATION_SECONDS:-0}"
INITIAL_MODEL="${ARMAGETRON_SELFPLAY_INITIAL_MODEL:-}"
FAST_MODE="${ARMAGETRON_SELFPLAY_FAST_MODE:-0}"
HEAVY_MODE="${ARMAGETRON_SELFPLAY_HEAVY_MODE:-0}"
THINK_TIME="${ARMAGETRON_SELFPLAY_THINK_TIME:-}"
TRAIN_EPOCHS="${ARMAGETRON_SELFPLAY_TRAIN_EPOCHS:-}"
DEDICATED_FPS_OVERRIDE="${ARMAGETRON_SELFPLAY_DEDICATED_FPS:-}"
MIN_PLAYERS_OVERRIDE="${ARMAGETRON_SELFPLAY_MIN_PLAYERS:-}"
TEAMS_MIN_OVERRIDE="${ARMAGETRON_SELFPLAY_TEAMS_MIN:-}"
TEAMS_MAX_OVERRIDE="${ARMAGETRON_SELFPLAY_TEAMS_MAX:-}"
PROFILE="${ARMAGETRON_SELFPLAY_PROFILE:-}"
PARENT_MODEL="${ARMAGETRON_SELFPLAY_PARENT_MODEL:-}"
BOT_COUNT_OVERRIDE="${ARMAGETRON_SELFPLAY_BOT_COUNT:-}"
EXPLORATION_OVERRIDE="${ARMAGETRON_SELFPLAY_EXPLORATION:-}"
POLICY_POOL_SIZE_OVERRIDE="${ARMAGETRON_SELFPLAY_POLICY_POOL_SIZE:-}"
POLICY_SNAPSHOT_EVERY_OVERRIDE="${ARMAGETRON_SELFPLAY_POLICY_SNAPSHOT_EVERY:-}"
POLICY_SNAPSHOT_WARMUP_OVERRIDE="${ARMAGETRON_SELFPLAY_POLICY_SNAPSHOT_WARMUP:-}"
POLICY_HISTORIC_PROB_OVERRIDE="${ARMAGETRON_SELFPLAY_POLICY_HISTORIC_PROB:-}"
BENCH_REPORT_ABS="${ARMAGETRON_SELFPLAY_BENCH_REPORT_ABS:-}"
CHOSEN_CHECKPOINT_ABS="${ARMAGETRON_SELFPLAY_CHOSEN_CHECKPOINT_ABS:-}"
PROMOTION_STATUS="${ARMAGETRON_SELFPLAY_PROMOTION_STATUS:-unreviewed}"
WORKER_COUNT="${ARMAGETRON_SELFPLAY_PARALLEL_WORKERS:-4}"
SYNC_SECONDS="${ARMAGETRON_SELFPLAY_SYNC_SECONDS:-300}"
PARALLEL_RECORD_STRIDE="${ARMAGETRON_SELFPLAY_PARALLEL_RECORD_STRIDE:-16}"

TRAINER_CFG_REL="${ARMAGETRON_SELFPLAY_TRAINER_CFG_REL:-generated_blacklight_parallel_trainer_${RUN_NAME}.cfg}"
TRAINER_CFG_PATH="${REPO_ROOT}/config/${TRAINER_CFG_REL}"
MANIFEST_PATH="${RUN_DIR_ABS}/run_manifest.env"
LATEST_MANIFEST_PATH="${BLACKLIGHT_LATEST_MANIFEST_PATH}"
PROGRESS_PATH="${REPO_ROOT}/var/${PROGRESS_REL}"
EVENTS_LOG_PATH="${REPO_ROOT}/var/${EVENTS_LOG_REL}"
TRAINER_LOG_PATH="${REPO_ROOT}/var/${TRAINER_LOG_REL}"

if [[ ! -f "${BASE_CFG_PATH}" ]]; then
    echo "Missing self-play config: ${BASE_CFG_PATH}" >&2
    exit 1
fi

if [[ "${FAST_MODE}" == "1" ]]; then
    echo "Parallel Blacklight training needs full worker experience logs, so --fast is not supported here." >&2
    exit 1
fi

if ! [[ "${WORKER_COUNT}" =~ ^[0-9]+$ ]] || (( WORKER_COUNT < 1 )); then
    echo "Parallel worker count must be a positive integer." >&2
    exit 1
fi

if ! [[ "${SYNC_SECONDS}" =~ ^[0-9]+$ ]] || (( SYNC_SECONDS < 1 )); then
    echo "Parallel sync seconds must be a positive integer." >&2
    exit 1
fi

if ! [[ "${PARALLEL_RECORD_STRIDE}" =~ ^[0-9]+$ ]] || (( PARALLEL_RECORD_STRIDE < 1 )); then
    echo "Parallel record stride must be a positive integer." >&2
    exit 1
fi

if [[ "${HEAVY_MODE}" == "1" ]]; then
    : "${THINK_TIME:=0.04}"
    : "${TRAIN_EPOCHS:=6}"
    : "${DEDICATED_FPS_OVERRIDE:=360}"
    : "${MIN_PLAYERS_OVERRIDE:=10}"
    : "${TEAMS_MIN_OVERRIDE:=10}"
    : "${TEAMS_MAX_OVERRIDE:=10}"
fi

effective_bot_count="${BOT_COUNT_OVERRIDE:-1}"
if [[ -n "${effective_bot_count}" && "${effective_bot_count}" =~ ^[0-9]+$ && effective_bot_count -le 1 ]]; then
    POLICY_POOL_SIZE_OVERRIDE="0"
    POLICY_HISTORIC_PROB_OVERRIDE="0"
    POLICY_SNAPSHOT_EVERY_OVERRIDE=""
    POLICY_SNAPSHOT_WARMUP_OVERRIDE=""
fi

mkdir -p "${RUN_DIR_ABS}/checkpoints"
mkdir -p "${RUN_DIR_ABS}/workers"
mkdir -p "${REPO_ROOT}/var/blacklight_runs"
mkdir -p "$(dirname "${TRAINER_CFG_PATH}")"

: > "${EVENTS_LOG_PATH}"
: > "${TRAINER_LOG_PATH}"

worker_name() {
    printf 'worker%02d' "$1"
}

worker_dir_rel() {
    printf '%s/workers/%s' "${RUN_DIR_REL}" "$(worker_name "$1")"
}

worker_dir_abs() {
    printf '%s/var/%s' "${REPO_ROOT}" "$(worker_dir_rel "$1")"
}

worker_model_rel() {
    printf '%s/trained_ai_model.txt' "$(worker_dir_rel "$1")"
}

worker_record_rel() {
    printf '%s/trained_ai_selfplay_experience.log' "$(worker_dir_rel "$1")"
}

worker_metrics_rel() {
    printf '%s/trained_ai_training_metrics.csv' "$(worker_dir_rel "$1")"
}

worker_console_log_rel() {
    printf '%s/console.log' "$(worker_dir_rel "$1")"
}

worker_console_log_abs() {
    printf '%s/var/%s' "${REPO_ROOT}" "$(worker_console_log_rel "$1")"
}

worker_cfg_rel() {
    printf 'generated_blacklight_parallel_%s_%s.cfg' "${RUN_NAME}" "$(worker_name "$1")"
}

worker_cfg_path() {
    printf '%s/config/%s' "${REPO_ROOT}" "$(worker_cfg_rel "$1")"
}

write_worker_cfg() {
    local worker_id="$1"
    local cfg_path
    cfg_path="$(worker_cfg_path "${worker_id}")"

    mkdir -p "$(worker_dir_abs "${worker_id}")"

    cat > "${cfg_path}" <<EOF
SINCLUDE ${BASE_CFG_REL}
SERVER_NAME Blacklight Parallel $(worker_name "${worker_id}")
AI_TRAINED_MODEL_FILE $(worker_model_rel "${worker_id}")
AI_TRAINED_RECORD_FILE $(worker_record_rel "${worker_id}")
AI_TRAINED_METRICS_FILE $(worker_metrics_rel "${worker_id}")
AI_TRAINED_LEARN 0
AI_TRAINED_RECORD 1
AI_TRAINED_RECORD_STRIDE ${PARALLEL_RECORD_STRIDE}
AI_TRAINED_SAVE_EVERY 1000000000
AI_TRAINED_CHECKPOINT_EVERY 1000000000
AI_TRAINED_OFFLINE_TRAIN 0
EOF

    if [[ -n "${BOT_COUNT_OVERRIDE}" ]]; then
        printf 'AI_TRAINED_BOT_COUNT %s\n' "${BOT_COUNT_OVERRIDE}" >> "${cfg_path}"
    fi

    if [[ -n "${EXPLORATION_OVERRIDE}" ]]; then
        printf 'AI_TRAINED_EXPLORATION %s\n' "${EXPLORATION_OVERRIDE}" >> "${cfg_path}"
    fi

    if [[ -n "${POLICY_POOL_SIZE_OVERRIDE}" ]]; then
        printf 'AI_TRAINED_POLICY_POOL_SIZE %s\n' "${POLICY_POOL_SIZE_OVERRIDE}" >> "${cfg_path}"
    else
        printf 'AI_TRAINED_POLICY_POOL_SIZE 0\n' >> "${cfg_path}"
    fi

    if [[ -n "${POLICY_SNAPSHOT_EVERY_OVERRIDE}" ]]; then
        printf 'AI_TRAINED_POLICY_SNAPSHOT_EVERY %s\n' "${POLICY_SNAPSHOT_EVERY_OVERRIDE}" >> "${cfg_path}"
    fi

    if [[ -n "${POLICY_SNAPSHOT_WARMUP_OVERRIDE}" ]]; then
        printf 'AI_TRAINED_POLICY_SNAPSHOT_WARMUP %s\n' "${POLICY_SNAPSHOT_WARMUP_OVERRIDE}" >> "${cfg_path}"
    fi

    if [[ -n "${POLICY_HISTORIC_PROB_OVERRIDE}" ]]; then
        printf 'AI_TRAINED_POLICY_HISTORIC_PROB %s\n' "${POLICY_HISTORIC_PROB_OVERRIDE}" >> "${cfg_path}"
    else
        printf 'AI_TRAINED_POLICY_HISTORIC_PROB 0\n' >> "${cfg_path}"
    fi

    if [[ -n "${THINK_TIME}" ]]; then
        printf 'AI_TRAINED_THINK_TIME %s\n' "${THINK_TIME}" >> "${cfg_path}"
    fi

    if [[ -n "${DEDICATED_FPS_OVERRIDE}" ]]; then
        printf 'DEDICATED_FPS %s\n' "${DEDICATED_FPS_OVERRIDE}" >> "${cfg_path}"
    fi

    if [[ -n "${MIN_PLAYERS_OVERRIDE}" ]]; then
        printf 'MIN_PLAYERS %s\n' "${MIN_PLAYERS_OVERRIDE}" >> "${cfg_path}"
    fi

    if [[ -n "${TEAMS_MIN_OVERRIDE}" ]]; then
        printf 'TEAMS_MIN %s\n' "${TEAMS_MIN_OVERRIDE}" >> "${cfg_path}"
    fi

    if [[ -n "${TEAMS_MAX_OVERRIDE}" ]]; then
        printf 'TEAMS_MAX %s\n' "${TEAMS_MAX_OVERRIDE}" >> "${cfg_path}"
    fi

    if [[ "${LIMIT_ROUNDS}" != "0" ]]; then
        printf 'LIMIT_ROUNDS %s\n' "${LIMIT_ROUNDS}" >> "${cfg_path}"
    fi
}

write_trainer_cfg() {
    cat > "${TRAINER_CFG_PATH}" <<EOF
SINCLUDE ${BASE_CFG_REL}
AI_TRAINED_MODEL_FILE ${MODEL_REL}
AI_TRAINED_METRICS_FILE ${METRICS_REL}
AI_TRAINED_CHECKPOINT_PREFIX ${CHECKPOINT_PREFIX_REL}
AI_TRAINED_RECORD 0
AI_TRAINED_OFFLINE_TRAIN 1
AI_TRAINED_OFFLINE_SOURCE_LIST ${SOURCE_LIST_REL}
AI_TRAINED_OFFLINE_STATE_FILE ${STATE_FILE_REL}
AI_TRAINED_POLICY_POOL_SIZE 0
AI_TRAINED_POLICY_HISTORIC_PROB 0
EOF

    if [[ -n "${CHECKPOINT_EVERY}" ]]; then
        printf 'AI_TRAINED_CHECKPOINT_EVERY %s\n' "${CHECKPOINT_EVERY}" >> "${TRAINER_CFG_PATH}"
    fi

    if [[ -n "${SAVE_EVERY}" ]]; then
        printf 'AI_TRAINED_SAVE_EVERY %s\n' "${SAVE_EVERY}" >> "${TRAINER_CFG_PATH}"
    fi

    if [[ -n "${TRAIN_EPOCHS}" ]]; then
        printf 'AI_TRAINED_TRAIN_EPOCHS %s\n' "${TRAIN_EPOCHS}" >> "${TRAINER_CFG_PATH}"
    fi
}

write_source_list() {
    local source_list_abs="${REPO_ROOT}/var/${SOURCE_LIST_REL}"
    : > "${source_list_abs}"
    local worker_id
    for (( worker_id = 1; worker_id <= WORKER_COUNT; ++worker_id )); do
        worker_record_rel "${worker_id}" >> "${source_list_abs}"
        printf '\n' >> "${source_list_abs}"
    done
}

cat > "${MANIFEST_PATH}" <<EOF
RUN_NAME=${RUN_NAME}
RUN_DIR_REL=${RUN_DIR_REL}
RUN_DIR_ABS=${RUN_DIR_ABS}
MODEL_REL=${MODEL_REL}
MODEL_ABS=${REPO_ROOT}/var/${MODEL_REL}
METRICS_REL=${METRICS_REL}
METRICS_ABS=${REPO_ROOT}/var/${METRICS_REL}
METRICS_SUMMARY_ABS=${REPO_ROOT}/var/${METRICS_REL}.latest
CHECKPOINT_PREFIX_REL=${CHECKPOINT_PREFIX_REL}
CHECKPOINT_PREFIX_ABS=${REPO_ROOT}/var/${CHECKPOINT_PREFIX_REL}
GENERATED_CFG_REL=${TRAINER_CFG_REL}
GENERATED_CFG_PATH=${TRAINER_CFG_PATH}
BASE_CFG_REL=${BASE_CFG_REL}
GENERATION=${GENERATION}
TRAINING_MODE=parallel
PROFILE=${PROFILE}
PARENT_MODEL=${PARENT_MODEL}
PARALLEL_WORKERS=${WORKER_COUNT}
SYNC_SECONDS=${SYNC_SECONDS}
PARALLEL_RECORD_STRIDE=${PARALLEL_RECORD_STRIDE}
SOURCE_LIST_REL=${SOURCE_LIST_REL}
SOURCE_LIST_ABS=${REPO_ROOT}/var/${SOURCE_LIST_REL}
STATE_FILE_REL=${STATE_FILE_REL}
STATE_FILE_ABS=${REPO_ROOT}/var/${STATE_FILE_REL}
PROGRESS_REL=${PROGRESS_REL}
PROGRESS_ABS=${PROGRESS_PATH}
EVENTS_LOG_REL=${EVENTS_LOG_REL}
EVENTS_LOG_ABS=${EVENTS_LOG_PATH}
TRAINER_LOG_REL=${TRAINER_LOG_REL}
TRAINER_LOG_ABS=${TRAINER_LOG_PATH}
DURATION_SECONDS=${DURATION_SECONDS}
BENCH_REPORT_ABS=${BENCH_REPORT_ABS}
CHOSEN_CHECKPOINT_ABS=${CHOSEN_CHECKPOINT_ABS}
PROMOTION_STATUS=${PROMOTION_STATUS}
EOF
cp "${MANIFEST_PATH}" "${LATEST_MANIFEST_PATH}"

if [[ -n "${INITIAL_MODEL}" ]]; then
    cp "${INITIAL_MODEL}" "${REPO_ROOT}/var/${MODEL_REL}"
fi

write_source_list
write_trainer_cfg

for (( worker_id = 1; worker_id <= WORKER_COUNT; ++worker_id )); do
    write_worker_cfg "${worker_id}"
done

if ! BIN_PATH="$(blacklight_find_server_bin "${ARMAGETRON_SELFPLAY_BIN:-}")"; then
    cat >&2 <<'EOF'
Could not find a dedicated-capable server binary.

Set ARMAGETRON_SELFPLAY_BIN to your dedicated binary path, then rerun:
  ARMAGETRON_SELFPLAY_BIN="/absolute/path/to/armagetronad-dedicated" ./scripts/blacklight.sh train --parallel-workers 4
EOF
    exit 1
fi
if ! blacklight_preflight_server_bin "${BIN_PATH}"; then
    exit 1
fi

COMMON_PREFIX_ARGS=(
    --datadir "${REPO_ROOT}"
    --configdir "${REPO_ROOT}/config"
    --userdatadir "${REPO_ROOT}/var"
    --vardir "${REPO_ROOT}/var"
)

worker_pid_list=""
trainer_pid=""
stop_requested=0
cleanup_done=0
current_phase="initializing"
current_cycle=0
training_started_at=0
training_end_at=0
cycle_deadline=0
sync_started_at=0
last_sync_started_at=0
last_sync_finished_at=0
last_sync_duration=0
last_sync_experience_bytes=0
sync_reason="initial"

append_event() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "${EVENTS_LOG_PATH}"
}

worker_pid_string() {
    printf '%s\n' "${worker_pid_list}"
}

update_progress_file() {
    cat > "${PROGRESS_PATH}" <<EOF
PHASE=${current_phase}
CURRENT_CYCLE=${current_cycle}
TRAINING_STARTED_AT=${training_started_at}
TRAINING_END_AT=${training_end_at}
CYCLE_DEADLINE=${cycle_deadline}
SYNC_STARTED_AT=${sync_started_at}
LAST_SYNC_STARTED_AT=${last_sync_started_at}
LAST_SYNC_FINISHED_AT=${last_sync_finished_at}
LAST_SYNC_DURATION=${last_sync_duration}
LAST_SYNC_EXPERIENCE_BYTES=${last_sync_experience_bytes}
WORKERS_RUNNING=$(blacklight_count_running_pids "$(worker_pid_string)")
WORKER_PIDS=$(worker_pid_string)
TRAINER_PID=${trainer_pid:-0}
SYNC_REASON=${sync_reason}
EVENTS_LOG_ABS=${EVENTS_LOG_PATH}
TRAINER_LOG_ABS=${TRAINER_LOG_PATH}
EOF
}

render_inline_status() {
    local message="$1"
    if [[ -t 1 ]]; then
        printf '\r\033[K%s' "${message}"
    fi
}

finish_inline_status() {
    local message="$1"
    if [[ -t 1 ]]; then
        printf '\r\033[K%s\n' "${message}"
    fi
}

current_experience_bytes() {
    local total=0
    local worker_id
    local path
    local size

    for (( worker_id = 1; worker_id <= WORKER_COUNT; ++worker_id )); do
        path="${REPO_ROOT}/var/$(worker_record_rel "${worker_id}")"
        size="$(blacklight_file_size "${path}")"
        total=$(( total + size ))
    done

    printf '%s\n' "${total}"
}

print_learning_summary() {
    local prefix="$1"
    local model_abs="${REPO_ROOT}/var/${MODEL_REL}"
    local summary_abs="${REPO_ROOT}/var/${METRICS_REL}.latest"
    local model_episodes="0"
    local model_updates="0"
    local metrics_episodes="0"
    local metrics_win_rate="n/a"
    local metrics_reward="n/a"

    if [[ -f "${model_abs}" ]]; then
        model_episodes="$(blacklight_model_episodes "${model_abs}")"
        model_updates="$(blacklight_model_updates "${model_abs}")"
    fi

    if [[ -f "${summary_abs}" ]]; then
        metrics_episodes="$(blacklight_read_summary_value "${summary_abs}" episodes)"
        metrics_win_rate="$(blacklight_read_summary_value "${summary_abs}" win_rate)"
        metrics_reward="$(blacklight_read_summary_value "${summary_abs}" average_reward)"
    fi

    printf '%smodel %s episodes / %s updates | metrics %s episodes | win rate %s | avg reward %s\n' \
        "${prefix}" \
        "${model_episodes:-0}" \
        "${model_updates:-0}" \
        "${metrics_episodes:-0}" \
        "${metrics_win_rate:-n/a}" \
        "${metrics_reward:-n/a}"
}

stop_trainer() {
    if [[ -n "${trainer_pid}" ]] && kill -0 "${trainer_pid}" 2>/dev/null; then
        kill -TERM "${trainer_pid}" 2>/dev/null || true
        wait "${trainer_pid}" 2>/dev/null || true
    fi
    trainer_pid=""
    update_progress_file
}

stop_workers() {
    local pid

    for pid in ${worker_pid_list}; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done

    for pid in ${worker_pid_list}; do
        wait "${pid}" 2>/dev/null || true
    done

    worker_pid_list=""
    update_progress_file
}

cleanup() {
    if [[ "${cleanup_done}" == "1" ]]; then
        return 0
    fi

    cleanup_done=1
    stop_workers
    stop_trainer
}

request_stop() {
    stop_requested=1
    current_phase="stopping"
    append_event "stop requested"
    update_progress_file
    cleanup
}

trap request_stop INT TERM
trap cleanup EXIT

run_trainer_sync() {
    local sync_pid=""
    local sync_status=0

    sync_reason="${1:-periodic}"
    sync_started_at="$(date +%s)"
    last_sync_started_at="${sync_started_at}"
    current_phase="syncing"
    cycle_deadline=0
    update_progress_file

    append_event "sync started reason=${sync_reason} cycle=${current_cycle}"
    printf '\nSyncing model (%s)...\n' "${sync_reason}"
    printf '  trainer log: %s\n' "${TRAINER_LOG_PATH}"

    {
        printf '[%s] sync-start reason=%s cycle=%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${sync_reason}" "${current_cycle}"
        "${BIN_PATH}" "${COMMON_PREFIX_ARGS[@]}" --extraconfig "${TRAINER_CFG_REL}"
    } >> "${TRAINER_LOG_PATH}" 2>&1 &
    trainer_pid="$!"
    sync_pid="${trainer_pid}"
    update_progress_file

    while kill -0 "${sync_pid}" 2>/dev/null; do
        local sync_elapsed
        local sync_bar

        sync_elapsed=$(( $(date +%s) - sync_started_at ))
        sync_bar="$(blacklight_render_activity_bar "${sync_elapsed}" 18)"
        render_inline_status "  phase: syncing model | cycle ${current_cycle} | ${sync_bar} | elapsed $(blacklight_format_duration "${sync_elapsed}")"
        update_progress_file
        sleep 1
    done

    if [[ -n "${trainer_pid}" ]]; then
        set +e
        wait "${sync_pid}"
        sync_status=$?
        set -e
    fi

    last_sync_finished_at="$(date +%s)"
    last_sync_duration=$(( last_sync_finished_at - sync_started_at ))
    trainer_pid=""
    sync_started_at=0
    if (( sync_status == 0 )); then
        last_sync_experience_bytes="$(current_experience_bytes)"
    fi
    update_progress_file

    if [[ "${stop_requested}" == "1" ]]; then
        finish_inline_status "  sync interrupted after $(blacklight_format_duration "${last_sync_duration}")"
        append_event "sync interrupted reason=${sync_reason} cycle=${current_cycle}"
        return 0
    fi

    if (( sync_status != 0 )); then
        finish_inline_status "  sync failed after $(blacklight_format_duration "${last_sync_duration}")"
        append_event "sync failed reason=${sync_reason} cycle=${current_cycle} exit=${sync_status}"
        echo "Trainer sync failed. See ${TRAINER_LOG_PATH}." >&2
        return "${sync_status}"
    fi

    finish_inline_status "  sync complete in $(blacklight_format_duration "${last_sync_duration}")"
    print_learning_summary "  snapshot: "
    append_event "sync complete reason=${sync_reason} cycle=${current_cycle} duration=${last_sync_duration}"
}

start_workers() {
    local worker_id
    local worker_model_abs
    local worker_console_log
    worker_pid_list=""
    current_phase="collecting"
    update_progress_file

    for (( worker_id = 1; worker_id <= WORKER_COUNT; ++worker_id )); do
        worker_model_abs="${REPO_ROOT}/var/$(worker_model_rel "${worker_id}")"
        worker_console_log="$(worker_console_log_abs "${worker_id}")"
        if [[ -f "${REPO_ROOT}/var/${MODEL_REL}" ]]; then
            cp "${REPO_ROOT}/var/${MODEL_REL}" "${worker_model_abs}"
        else
            rm -f "${worker_model_abs}"
        fi

        printf '[%s] cycle=%s worker=%s start\n' \
            "$(date '+%Y-%m-%d %H:%M:%S')" \
            "${current_cycle}" \
            "$(worker_name "${worker_id}")" >> "${worker_console_log}"
        "${BIN_PATH}" "${COMMON_PREFIX_ARGS[@]}" --extraconfig "$(worker_cfg_rel "${worker_id}")" >> "${worker_console_log}" 2>&1 &
        if [[ -n "${worker_pid_list}" ]]; then
            worker_pid_list="${worker_pid_list} "
        fi
        worker_pid_list="${worker_pid_list}$!"
    done

    update_progress_file
}

wait_for_sync_window() {
    local cycle_deadline="$1"
    local now
    local running_workers
    local experience_bytes
    local cycle_start_at
    local cycle_total_seconds
    local cycle_elapsed_seconds
    local cycle_bar
    local worker_bar

    cycle_start_at="${last_sync_finished_at:-${training_started_at}}"

    while :; do
        if [[ "${stop_requested}" == "1" ]]; then
            return 1
        fi

        running_workers="$(blacklight_count_running_pids "$(worker_pid_string)")"

        if [[ "${running_workers}" == "0" ]]; then
            finish_inline_status "  workers exited early"
            return 0
        fi

        now="$(date +%s)"
        if (( now >= cycle_deadline )); then
            finish_inline_status "  worker window complete"
            return 0
        fi

        experience_bytes="$(current_experience_bytes)"
        cycle_total_seconds=$(( cycle_deadline - cycle_start_at ))
        cycle_elapsed_seconds=$(( now - cycle_start_at ))
        cycle_bar="$(blacklight_render_progress_bar "${cycle_elapsed_seconds}" "${cycle_total_seconds}" 18)"
        worker_bar="$(blacklight_render_progress_bar "${running_workers}" "${WORKER_COUNT}" 10)"
        render_inline_status "  phase: collecting | cycle ${current_cycle} | window ${cycle_bar} | workers ${worker_bar} ${running_workers}/${WORKER_COUNT} | remaining $(blacklight_format_duration $(( cycle_deadline - now ))) | experience $(blacklight_human_bytes "${experience_bytes}")"
        update_progress_file
        sleep 1
    done
}

update_progress_file

echo "Starting Blacklight parallel trainer..."
echo "Binary: ${BIN_PATH}"
echo "Run: ${RUN_NAME}"
echo "Run dir: ${RUN_DIR_ABS}"
echo "Manifest: ${MANIFEST_PATH}"
echo "Trainer config: ${TRAINER_CFG_REL}"
echo "Parallel workers: ${WORKER_COUNT}"
echo "Sync window: ${SYNC_SECONDS}s"
echo "Record stride: ${PARALLEL_RECORD_STRIDE}"
echo "Monitor: ./scripts/blacklight.sh monitor"
echo "Events log: ${EVENTS_LOG_PATH}"
echo "Trainer log: ${TRAINER_LOG_PATH}"
echo "Worker logs: ${RUN_DIR_ABS}/workers/workerXX/console.log"
if [[ "${HEAVY_MODE}" == "1" ]]; then
    echo "Heavy mode: on (faster thinking, more epochs, higher FPS, bigger rooms)"
fi

run_trainer_sync initial

start_timestamp="$(date +%s)"
end_timestamp=0
training_started_at="${start_timestamp}"
if [[ "${DURATION_SECONDS}" != "0" ]]; then
    end_timestamp=$(( start_timestamp + DURATION_SECONDS ))
    training_end_at="${end_timestamp}"
    echo "Duration: ${DURATION_SECONDS}s ($(blacklight_format_duration "${DURATION_SECONDS}"))"
else
    training_end_at=0
    echo "Stop with Ctrl+C."
fi
update_progress_file

while :; do
    if [[ "${stop_requested}" == "1" ]]; then
        break
    fi

    current_timestamp="$(date +%s)"
    if (( end_timestamp > 0 && current_timestamp >= end_timestamp )); then
        break
    fi

    current_cycle=$(( current_cycle + 1 ))
    cycle_deadline=$(( current_timestamp + SYNC_SECONDS ))
    if (( end_timestamp > 0 && cycle_deadline > end_timestamp )); then
        cycle_deadline="${end_timestamp}"
    fi

    append_event "cycle started cycle=${current_cycle} workers=${WORKER_COUNT}"
    echo
    echo "Cycle ${current_cycle}: collecting experience from ${WORKER_COUNT} workers..."
    start_workers
    wait_for_sync_window "${cycle_deadline}" || true
    stop_workers

    if [[ "${stop_requested}" == "1" ]]; then
        break
    fi

    run_trainer_sync periodic
done

trap - INT TERM
stop_workers
current_phase="finished"
cycle_deadline=0
sync_started_at=0
update_progress_file
append_event "training finished"

echo "Blacklight parallel training run finished."
