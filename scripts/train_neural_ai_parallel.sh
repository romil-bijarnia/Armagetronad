#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/blacklight_lib.sh"
REPO_ROOT="${BLACKLIGHT_REPO_ROOT}"

BASE_CFG_REL="${ARMAGETRON_SELFPLAY_BASE_CFG_REL:-examples/trained_ai_teacher_collection.cfg}"
BASE_CFG_PATH="${REPO_ROOT}/config/${BASE_CFG_REL}"
GENERATION="${ARMAGETRON_SELFPLAY_GENERATION:-blacklight_teacher}"
RUN_ID="${ARMAGETRON_SELFPLAY_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${ARMAGETRON_SELFPLAY_RUN_NAME:-${GENERATION}_${RUN_ID}}"
RUN_DIR_REL="${ARMAGETRON_SELFPLAY_RUN_DIR_REL:-blacklight_runs/${RUN_NAME}}"
RUN_DIR_ABS="${REPO_ROOT}/var/${RUN_DIR_REL}"

MODEL_REL="${ARMAGETRON_SELFPLAY_MODEL_REL:-${RUN_DIR_REL}/trained_ai_teacher_cnn_model.txt}"
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
REPLAY_ONLY="${ARMAGETRON_SELFPLAY_REPLAY_ONLY:-0}"
REPLAY_SOURCE_LIST_ABS="${ARMAGETRON_SELFPLAY_REPLAY_SOURCE_LIST_ABS:-}"
COLLECT_ONLY="${ARMAGETRON_SELFPLAY_COLLECT_ONLY:-0}"

TRAINER_CFG_REL="${ARMAGETRON_SELFPLAY_TRAINER_CFG_REL:-generated_blacklight_parallel_trainer_${RUN_NAME}.cfg}"
TRAINER_CFG_PATH="${REPO_ROOT}/config/${TRAINER_CFG_REL}"
MANIFEST_PATH="${RUN_DIR_ABS}/run_manifest.env"
LATEST_MANIFEST_PATH="${BLACKLIGHT_LATEST_MANIFEST_PATH}"
PROGRESS_PATH="${REPO_ROOT}/var/${PROGRESS_REL}"
EVENTS_LOG_PATH="${REPO_ROOT}/var/${EVENTS_LOG_REL}"
TRAINER_LOG_PATH="${REPO_ROOT}/var/${TRAINER_LOG_REL}"

if [[ ! -f "${BASE_CFG_PATH}" ]]; then
    echo "Missing teacher collection config: ${BASE_CFG_PATH}" >&2
    exit 1
fi

if [[ "${FAST_MODE}" == "1" ]]; then
    echo "Teacher-imitation training needs full worker teacher logs, so --fast is not supported here." >&2
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

if [[ "${REPLAY_ONLY}" == "1" && -n "${REPLAY_SOURCE_LIST_ABS}" && ! -f "${REPLAY_SOURCE_LIST_ABS}" ]]; then
    echo "Replay source list does not exist: ${REPLAY_SOURCE_LIST_ABS}" >&2
    exit 1
fi

if [[ "${REPLAY_ONLY}" == "1" && -z "${REPLAY_SOURCE_LIST_ABS}" ]]; then
    echo "Replay mode requires ARMAGETRON_SELFPLAY_REPLAY_SOURCE_LIST_ABS." >&2
    exit 1
fi

if [[ "${REPLAY_ONLY}" == "1" && "${COLLECT_ONLY}" == "1" ]]; then
    echo "Replay mode and collect-only mode cannot be enabled together." >&2
    exit 1
fi

if [[ "${REPLAY_ONLY}" == "1" ]]; then
    : "${CHECKPOINT_EVERY:=0}"
    : "${SAVE_EVERY:=1000}"
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
    printf '%s/trained_ai_teacher_cnn_model.txt' "$(worker_dir_rel "$1")"
}

worker_experience_rel() {
    printf '%s/trained_ai_experience.log' "$(worker_dir_rel "$1")"
}

worker_record_rel() {
    printf '%s/trained_ai_teacher_examples.log' "$(worker_dir_rel "$1")"
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
SERVER_NAME Blacklight Teacher $(worker_name "${worker_id}")
AI_TRAINED_ENABLE 1
AI_TRAINED_AUTOSTART 1
AI_TRAINED_BOT_COUNT 1
AI_TRAINED_MODEL_FILE $(worker_model_rel "${worker_id}")
AI_TRAINED_RECORD_FILE $(worker_experience_rel "${worker_id}")
AI_TRAINED_TEACHER_FILE $(worker_record_rel "${worker_id}")
AI_TRAINED_LEARN 0
AI_TRAINED_RECORD 1
AI_TRAINED_RECORD_STRIDE ${PARALLEL_RECORD_STRIDE}
AI_TRAINED_SAVE_EVERY 1000000000
AI_TRAINED_CHECKPOINT_EVERY 1000000000
AI_TRAINED_OFFLINE_TRAIN 0
EOF

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
AI_TRAINED_ENABLE 0
AI_TRAINED_LEARN 0
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
    if [[ "${REPLAY_ONLY}" == "1" && -n "${REPLAY_SOURCE_LIST_ABS}" ]]; then
        cp "${REPLAY_SOURCE_LIST_ABS}" "${source_list_abs}"
        return 0
    fi

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
TRAINING_MODE=teacher
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
REPLAY_ONLY=${REPLAY_ONLY}
REPLAY_SOURCE_LIST_ABS=${REPLAY_SOURCE_LIST_ABS}
COLLECT_ONLY=${COLLECT_ONLY}
EOF
cp "${MANIFEST_PATH}" "${LATEST_MANIFEST_PATH}"

if [[ -n "${INITIAL_MODEL}" ]]; then
    cp "${INITIAL_MODEL}" "${REPO_ROOT}/var/${MODEL_REL}"
fi

write_source_list
write_trainer_cfg

if [[ "${REPLAY_ONLY}" != "1" ]]; then
    for (( worker_id = 1; worker_id <= WORKER_COUNT; ++worker_id )); do
        write_worker_cfg "${worker_id}"
    done
fi

if ! BIN_PATH="$(blacklight_find_server_bin "${ARMAGETRON_SELFPLAY_BIN:-}")"; then
    cat >&2 <<'EOF'
Could not find a dedicated-capable server binary.

Set ARMAGETRON_SELFPLAY_BIN to your dedicated binary path, then rerun:
  ARMAGETRON_SELFPLAY_BIN="/absolute/path/to/armagetronad-dedicated" ./scripts/blacklight.sh collect --parallel-workers 4
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
    {
        printf 'PHASE=%q\n' "${current_phase}"
        printf 'CURRENT_CYCLE=%q\n' "${current_cycle}"
        printf 'TRAINING_STARTED_AT=%q\n' "${training_started_at}"
        printf 'TRAINING_END_AT=%q\n' "${training_end_at}"
        printf 'CYCLE_DEADLINE=%q\n' "${cycle_deadline}"
        printf 'SYNC_STARTED_AT=%q\n' "${sync_started_at}"
        printf 'LAST_SYNC_STARTED_AT=%q\n' "${last_sync_started_at}"
        printf 'LAST_SYNC_FINISHED_AT=%q\n' "${last_sync_finished_at}"
        printf 'LAST_SYNC_DURATION=%q\n' "${last_sync_duration}"
        printf 'LAST_SYNC_EXPERIENCE_BYTES=%q\n' "${last_sync_experience_bytes}"
        printf 'WORKERS_RUNNING=%q\n' "$(blacklight_count_running_pids "$(worker_pid_string)")"
        printf 'WORKER_PIDS=%q\n' "$(worker_pid_string)"
        printf 'TRAINER_PID=%q\n' "${trainer_pid:-0}"
        printf 'SYNC_REASON=%q\n' "${sync_reason}"
        printf 'EVENTS_LOG_ABS=%q\n' "${EVENTS_LOG_PATH}"
        printf 'TRAINER_LOG_ABS=%q\n' "${TRAINER_LOG_PATH}"
    } > "${PROGRESS_PATH}"
}

render_inline_status() {
    local message="$1"
    if [[ -t 1 ]]; then
        printf '\r\033[2K%s' "$(fit_terminal_line "${message}")"
    fi
}

finish_inline_status() {
    local message="$1"
    if [[ -t 1 ]]; then
        printf '\r\033[2K%s\n' "$(fit_terminal_line "${message}")"
    fi
}

terminal_width() {
    local width="${COLUMNS:-}"

    if [[ -t 1 ]] && command -v tput >/dev/null 2>&1; then
        width="$(tput cols 2>/dev/null || printf '%s' "${width:-}")"
    fi

    if [[ -z "${width}" || ! "${width}" =~ ^[0-9]+$ ]] || (( width < 48 )); then
        width=80
    fi

    printf '%s\n' "${width}"
}

trim_terminal_text() {
    local text="$1"
    local max_len="$2"

    if [[ -z "${max_len}" || ! "${max_len}" =~ ^[0-9]+$ ]] || (( max_len <= 0 )); then
        printf '%s' "${text}"
        return 0
    fi

    if (( ${#text} <= max_len )); then
        printf '%s' "${text}"
        return 0
    fi

    if (( max_len <= 3 )); then
        printf '%.*s' "${max_len}" "${text}"
        return 0
    fi

    printf '%.*s...' "$(( max_len - 3 ))" "${text}"
}

fit_terminal_line() {
    local message="$1"
    local width
    local max_len

    width="$(terminal_width)"
    max_len=$(( width - 1 ))
    if (( max_len < 20 )); then
        max_len=20
    fi

    trim_terminal_text "${message}" "${max_len}"
}

progress_bar_width() {
    local wide="$1"
    local narrow="$2"
    local width

    width="$(terminal_width)"
    if (( width < 96 )); then
        printf '%s\n' "${narrow}"
    else
        printf '%s\n' "${wide}"
    fi
}

status_unicode() {
    [[ "${BLACKLIGHT_TRAIN_ASCII:-0}" != "1" ]]
}

status_repeat() {
    local char="$1"
    local count="${2:-0}"
    local index

    if [[ -z "${count}" || ! "${count}" =~ ^[0-9]+$ ]] || (( count <= 0 )); then
        return 0
    fi

    for (( index = 0; index < count; ++index )); do
        printf '%s' "${char}"
    done
}

status_progress_meter() {
    local current="${1:-0}"
    local total="${2:-0}"
    local width="${3:-18}"
    local filled=0
    local index

    if ! status_unicode; then
        blacklight_render_progress_bar "${current}" "${total}" "${width}" "#" "."
        return 0
    fi

    if [[ -z "${width}" || ! "${width}" =~ ^[0-9]+$ ]] || (( width < 1 )); then
        width=18
    fi
    if [[ -z "${current}" || ! "${current}" =~ ^-?[0-9]+$ ]]; then
        current=0
    fi
    if [[ -z "${total}" || ! "${total}" =~ ^[0-9]+$ ]] || (( total <= 0 )); then
        total=1
    fi
    if (( current < 0 )); then
        current=0
    elif (( current > total )); then
        current="${total}"
    fi

    filled=$(( (current * width + total / 2) / total ))
    if (( filled > width )); then
        filled="${width}"
    fi

    printf '▕'
    for (( index = 0; index < width; ++index )); do
        if (( index < filled )); then
            printf '█'
        else
            printf '░'
        fi
    done
    printf '▏'
}

status_activity_meter() {
    local tick="${1:-0}"
    local width="${2:-16}"
    local index
    local position

    if ! status_unicode; then
        blacklight_render_activity_bar "${tick}" "${width}"
        return 0
    fi

    if [[ -z "${tick}" || ! "${tick}" =~ ^-?[0-9]+$ ]]; then
        tick=0
    fi
    if [[ -z "${width}" || ! "${width}" =~ ^[0-9]+$ ]] || (( width < 1 )); then
        width=16
    fi
    if (( tick < 0 )); then
        tick=$(( -tick ))
    fi
    position=$(( tick % width ))

    printf '▕'
    for (( index = 0; index < width; ++index )); do
        if (( index == position )); then
            printf '◆'
        else
            printf '─'
        fi
    done
    printf '▏'
}

collect_status_line() {
    local cycle="$1"
    local cycle_bar="$2"
    local worker_bar="$3"
    local running_workers="$4"
    local remaining="$5"
    local experience_bytes="$6"

    printf 'BLACKLIGHT COLLECT  c%-3s %s  eta %s  workers %s %s/%s  data %s' \
        "${cycle}" \
        "${cycle_bar}" \
        "${remaining}" \
        "${worker_bar}" \
        "${running_workers}" \
        "${WORKER_COUNT}" \
        "$(blacklight_human_bytes "${experience_bytes}")"
}

sync_status_line() {
    local reason="$1"
    local cycle="$2"
    local sync_bar="$3"
    local elapsed="$4"

    printf 'BLACKLIGHT TRAIN    %-8s c%-3s %s  elapsed %s' \
        "${reason}" \
        "${cycle}" \
        "${sync_bar}" \
        "${elapsed}"
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
    local metrics_policy_loss="n/a"
    local metrics_value_loss="n/a"
    local metrics_entropy="n/a"
    local metrics_learned_episodes="n/a"
    local metrics_steps="n/a"
    local line_one=""
    local line_two=""

    if [[ -f "${model_abs}" ]]; then
        model_episodes="$(blacklight_model_episodes "${model_abs}")"
        model_updates="$(blacklight_model_updates "${model_abs}")"
    fi

    if [[ -f "${summary_abs}" ]]; then
        metrics_episodes="$(blacklight_read_summary_value "${summary_abs}" episodes)"
        metrics_policy_loss="$(blacklight_read_summary_value "${summary_abs}" average_policy_loss)"
        metrics_value_loss="$(blacklight_read_summary_value "${summary_abs}" average_value_loss)"
        metrics_entropy="$(blacklight_read_summary_value "${summary_abs}" average_entropy)"
        metrics_learned_episodes="$(blacklight_read_summary_value "${summary_abs}" learned_episodes)"
        metrics_steps="$(blacklight_read_summary_value "${summary_abs}" average_steps)"
    fi

    if [[ "${metrics_episodes:-0}" == "0" ]]; then
        metrics_steps="n/a"
    fi
    if [[ "${metrics_learned_episodes:-0}" == "0" ]]; then
        metrics_policy_loss="n/a"
        metrics_value_loss="n/a"
        metrics_entropy="n/a"
    fi

    line_one="$(printf '%smodel %s ep / %s upd  teacher %s ep  learned %s' \
        "${prefix}" \
        "${model_episodes:-0}" \
        "${model_updates:-0}" \
        "${metrics_episodes:-0}" \
        "${metrics_learned_episodes:-n/a}")"
    line_two="$(printf '  losses  policy %s  value %s  entropy %s  steps/ep %s' \
        "${metrics_policy_loss:-n/a}" \
        "${metrics_value_loss:-n/a}" \
        "${metrics_entropy:-n/a}" \
        "${metrics_steps:-n/a}"
    )"

    printf '%s\n' "$(fit_terminal_line "${line_one}")"
    printf '%s\n' "$(fit_terminal_line "${line_two}")"
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
    if status_unicode; then
        printf '\n╭─ Blacklight training sync: %s\n' "${sync_reason}"
        printf '│ trainer log: %s\n' "${TRAINER_LOG_PATH}"
    else
        printf '\nBlacklight training sync: %s\n' "${sync_reason}"
        printf '  trainer log: %s\n' "${TRAINER_LOG_PATH}"
    fi

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
        sync_bar="$(status_activity_meter "${sync_elapsed}" "$(progress_bar_width 18 10)")"
        render_inline_status "$(sync_status_line "${sync_reason}" "${current_cycle}" "${sync_bar}" "$(blacklight_format_duration "${sync_elapsed}")")"
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
        cycle_bar="$(status_progress_meter "${cycle_elapsed_seconds}" "${cycle_total_seconds}" "$(progress_bar_width 20 12)")"
        worker_bar="$(status_progress_meter "${running_workers}" "${WORKER_COUNT}" "$(progress_bar_width 10 6)")"
        render_inline_status "$(collect_status_line "${current_cycle}" "${cycle_bar}" "${worker_bar}" "${running_workers}" "$(blacklight_format_duration $(( cycle_deadline - now )))" "${experience_bytes}")"
        update_progress_file
        sleep 1
    done
}

update_progress_file

echo "Starting Blacklight teacher trainer..."
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
if [[ "${REPLAY_ONLY}" == "1" ]]; then
    echo "Replay source list: ${REPLAY_SOURCE_LIST_ABS}"
fi
if [[ "${COLLECT_ONLY}" == "1" ]]; then
    echo "Collect-only mode: on"
fi
if [[ "${HEAVY_MODE}" == "1" ]]; then
    echo "Heavy mode: on (faster thinking, more epochs, higher FPS, bigger rooms)"
fi

start_timestamp="$(date +%s)"
end_timestamp=0
training_started_at="${start_timestamp}"
if [[ "${REPLAY_ONLY}" == "1" ]]; then
    training_end_at="${start_timestamp}"
    update_progress_file
    run_trainer_sync replay
    trap - INT TERM
    stop_workers
    current_phase="finished"
    cycle_deadline=0
    sync_started_at=0
    update_progress_file
    append_event "training finished"
    echo "Blacklight replay training run finished."
    exit 0
fi

if [[ "${COLLECT_ONLY}" == "1" ]]; then
    if [[ "${DURATION_SECONDS}" != "0" ]]; then
        end_timestamp=$(( start_timestamp + DURATION_SECONDS ))
        training_end_at="${end_timestamp}"
        echo "Collection duration: ${DURATION_SECONDS}s ($(blacklight_format_duration "${DURATION_SECONDS}"))"
    else
        training_end_at=0
        echo "Collecting fresh teacher data. Stop with Ctrl+C."
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

        append_event "collect cycle started cycle=${current_cycle} workers=${WORKER_COUNT}"
        echo
        echo "Cycle ${current_cycle}: collecting teacher examples from ${WORKER_COUNT} workers..."
        start_workers
        wait_for_sync_window "${cycle_deadline}" || true
        stop_workers
    done

    trap - INT TERM
    stop_workers
    current_phase="finished"
    cycle_deadline=0
    sync_started_at=0
    update_progress_file
    append_event "collection finished"
    echo "Blacklight teacher data collection finished."
    exit 0
fi

run_trainer_sync initial

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
    echo "Cycle ${current_cycle}: collecting teacher examples from ${WORKER_COUNT} workers..."
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
