#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/blacklight_lib.sh"
REPO_ROOT="${BLACKLIGHT_REPO_ROOT}"

usage() {
    cat <<'EOF'
Blacklight helper

Usage:
  ./scripts/blacklight.sh train [options]
  ./scripts/blacklight.sh smoke [options]
  ./scripts/blacklight.sh eval [options]
  ./scripts/blacklight.sh bench [options]
  ./scripts/blacklight.sh promote [options]
  ./scripts/blacklight.sh sweep [options]
  ./scripts/blacklight.sh status
  ./scripts/blacklight.sh monitor [options]
  ./scripts/blacklight.sh help

If no command is given, "train" is used.

Train options:
  --profile NAME
  --duration SECONDS
  --rounds COUNT
  --checkpoint-every COUNT
  --resume MODEL_PATH_OR_RUN
  --generation NAME
  --name RUN_NAME
  --bin PATH
  --fast
  --heavy
  --parallel-workers COUNT
  --sync-seconds SECONDS

Smoke options:
  --duration SECONDS
  --rounds COUNT
  --bin PATH

Eval options:
  --duration SECONDS
  --rounds COUNT
  --candidate MODEL_PATH
  --reference MODEL_PATH
  --bin PATH

Bench options:
  --suite NAME
  --candidate MODEL_PATH_OR_RUN
  --reference MODEL_PATH_OR_RUN
  --bin PATH

Promote options:
  --candidate MODEL_PATH_OR_RUN
  --bin PATH

Sweep options:
  --profile-chain classic
  --bin PATH

Monitor options:
  --interval SECONDS
  --once
EOF
}

require_latest_manifest() {
    blacklight_require_latest_manifest || exit 1
}

read_summary_value() {
    local summary_path="$1"
    local key="$2"
    blacklight_read_summary_value "${summary_path}" "${key}"
}

load_current_run_state() {
    require_latest_manifest
    blacklight_source_latest_manifest
    blacklight_source_progress || true
}

phase_label() {
    case "${1:-}" in
        initializing)
            printf 'Initializing\n'
            ;;
        collecting)
            printf 'Collecting experience\n'
            ;;
        syncing)
            printf 'Syncing model\n'
            ;;
        stopping)
            printf 'Stopping workers\n'
            ;;
        finished)
            printf 'Finished\n'
            ;;
        *)
            printf 'Unknown\n'
            ;;
    esac
}

worker_experience_bytes() {
    local total=0
    local record_rel
    local record_abs
    local record_size

    if [[ -f "${RECORD_ABS:-}" ]]; then
        blacklight_file_size "${RECORD_ABS}"
        return 0
    fi

    if [[ ! -f "${SOURCE_LIST_ABS:-}" ]]; then
        printf '0\n'
        return 0
    fi

    while IFS= read -r record_rel; do
        [[ -n "${record_rel}" ]] || continue
        record_abs="${REPO_ROOT}/var/${record_rel}"
        record_size="$(blacklight_file_size "${record_abs}")"
        total=$(( total + record_size ))
    done < "${SOURCE_LIST_ABS}"

    printf '%s\n' "${total}"
}

running_worker_count() {
    if [[ -n "${WORKER_PIDS:-}" ]]; then
        # shellcheck disable=SC2086
        blacklight_count_running_pids ${WORKER_PIDS}
        return 0
    fi

    if [[ "${TRAINING_MODE:-}" == "parallel" ]]; then
        ps -axo command= | awk -v pattern="generated_blacklight_parallel_${RUN_NAME:-}_worker" 'index($0, pattern) { count++ } END { print count + 0 }'
        return 0
    fi

    printf '%s\n' "${WORKERS_RUNNING:-0}"
}

runtime_phase() {
    local trainer_count
    local worker_count

    if [[ -n "${PHASE:-}" ]]; then
        printf '%s\n' "${PHASE}"
        return 0
    fi

    if [[ "${TRAINING_MODE:-}" == "parallel" ]]; then
        trainer_count="$(ps -axo command= | awk -v pattern="--extraconfig ${GENERATED_CFG_REL:-}" 'index($0, pattern) { count++ } END { print count + 0 }')"
        if (( trainer_count > 0 )); then
            printf 'syncing\n'
            return 0
        fi

        worker_count="$(running_worker_count)"
        if (( worker_count > 0 )); then
            printf 'collecting\n'
            return 0
        fi
    fi

    printf 'unknown\n'
}

print_status_line() {
    local label="$1"
    local value="$2"
    printf '%-20s %s\n' "${label}" "${value}"
}

print_source_progress_status() {
    if [[ -z "${STATE_FILE_ABS:-}" || -z "${SOURCE_LIST_ABS:-}" ]]; then
        return 0
    fi

    while IFS='|' read -r source_path source_episodes source_offset; do
        [[ -n "${source_path}" ]] || continue
        echo "source_progress ${source_path} episodes=${source_episodes} offset_bytes=${source_offset}"
    done < <(blacklight_format_source_progress "${STATE_FILE_ABS}" "${SOURCE_LIST_ABS}")
}

show_status() {
    load_current_run_state

    local model_stats=""
    local model_episodes="0"
    local model_updates="0"
    local latest_checkpoint_file="${CHECKPOINT_PREFIX_ABS:-}_latest.txt"
    local latest_checkpoint=""
    local latest_eval_report=""
    local current_phase=""
    local current_cycle=""
    local elapsed_display="n/a"
    local remaining_display="n/a"
    local running_workers="0"
    local worker_bytes="0"
    local experience_growth_bytes="0"
    local last_sync_display=""
    local avg_sync_seconds="n/a"
    local now=""
    local runtime_phase_value=""
    local rolling100_win_rate="n/a"
    local rolling100_reward="n/a"
    local rolling100_distance="n/a"
    local rolling500_win_rate="n/a"
    local rolling500_reward="n/a"
    local rolling500_distance="n/a"
    local champion_model=""

    if [[ -f "${MODEL_ABS:-}" ]]; then
        model_stats="$(blacklight_model_stats_line "${MODEL_ABS}")"
        model_episodes="$(printf '%s\n' "${model_stats}" | awk '{ print $2 }')"
        model_updates="$(printf '%s\n' "${model_stats}" | awk '{ print $3 }')"
    fi

    if [[ -f "${latest_checkpoint_file}" ]]; then
        local checkpoint_rel
        checkpoint_rel="$(sed -n '1p' "${latest_checkpoint_file}")"
        if [[ -n "${checkpoint_rel}" ]]; then
            latest_checkpoint="${REPO_ROOT}/var/${checkpoint_rel}"
        fi
    fi

    latest_eval_report="$(blacklight_latest_eval_report)"
    runtime_phase_value="$(runtime_phase)"
    current_phase="$(phase_label "${runtime_phase_value}")"
    current_cycle="${CURRENT_CYCLE:-0}"
    running_workers="$(running_worker_count)"
    worker_bytes="$(worker_experience_bytes)"
    experience_growth_bytes="$(blacklight_current_experience_growth "${worker_bytes}" "${LAST_SYNC_EXPERIENCE_BYTES:-0}")"
    avg_sync_seconds="$(blacklight_read_sync_metric "${EVENTS_LOG_ABS:-/dev/null}" avg)"

    if [[ -f "${METRICS_ABS:-}" ]]; then
        rolling100_win_rate="$(blacklight_read_metric_window "${METRICS_ABS}" 100 2)"
        rolling100_reward="$(blacklight_read_metric_window "${METRICS_ABS}" 100 3)"
        rolling100_distance="$(blacklight_read_metric_window "${METRICS_ABS}" 100 4)"
        rolling500_win_rate="$(blacklight_read_metric_window "${METRICS_ABS}" 500 2)"
        rolling500_reward="$(blacklight_read_metric_window "${METRICS_ABS}" 500 3)"
        rolling500_distance="$(blacklight_read_metric_window "${METRICS_ABS}" 500 4)"
    fi

    champion_model="$(blacklight_current_champion_model || true)"

    now="$(date +%s)"
    if [[ -n "${TRAINING_STARTED_AT:-}" && "${TRAINING_STARTED_AT}" =~ ^[0-9]+$ ]]; then
        elapsed_display="$(blacklight_format_duration $(( now - TRAINING_STARTED_AT )))"
    fi
    if [[ "${runtime_phase_value}" == "finished" ]]; then
        remaining_display="00:00:00"
    elif [[ -n "${TRAINING_END_AT:-}" && "${TRAINING_END_AT}" =~ ^[0-9]+$ ]]; then
        remaining_display="$(blacklight_format_duration $(( TRAINING_END_AT - now )))"
    fi
    if [[ -n "${LAST_SYNC_DURATION:-}" && "${LAST_SYNC_DURATION}" =~ ^[0-9]+$ ]]; then
        last_sync_display="$(blacklight_format_duration "${LAST_SYNC_DURATION}")"
    fi

    echo "Blacklight status"
    echo "run_name ${RUN_NAME:-unknown}"
    echo "generation ${GENERATION:-unknown}"
    if [[ -n "${PROFILE:-}" ]]; then
        echo "profile ${PROFILE}"
    fi
    if [[ -n "${TRAINING_MODE:-}" ]]; then
        echo "training_mode ${TRAINING_MODE}"
    fi
    if [[ -n "${PARENT_MODEL:-}" ]]; then
        echo "parent_model ${PARENT_MODEL}"
    fi
    if [[ -n "${PARALLEL_WORKERS:-}" ]]; then
        echo "parallel_workers ${PARALLEL_WORKERS}"
    fi
    if [[ -n "${SYNC_SECONDS:-}" ]]; then
        echo "sync_seconds ${SYNC_SECONDS}"
    fi
    echo "run_dir ${RUN_DIR_ABS:-unknown}"
    echo "model_file ${MODEL_ABS:-missing}"
    echo "model_episodes ${model_episodes:-0}"
    echo "model_updates ${model_updates:-0}"
    echo "phase ${runtime_phase_value}"
    echo "phase_label ${current_phase}"
    echo "current_cycle ${current_cycle}"
    echo "workers_running ${running_workers}"
    echo "experience_bytes ${worker_bytes}"
    echo "experience_growth_bytes ${experience_growth_bytes}"
    echo "elapsed ${elapsed_display}"
    echo "remaining ${remaining_display}"
    if [[ -n "${last_sync_display}" ]]; then
        echo "last_sync_duration ${last_sync_display}"
    fi
    echo "avg_sync_seconds ${avg_sync_seconds}"
    if [[ -n "${PROGRESS_ABS:-}" ]]; then
        echo "progress_file ${PROGRESS_ABS}"
    fi
    if [[ -n "${TRAINER_LOG_ABS:-}" ]]; then
        echo "trainer_log ${TRAINER_LOG_ABS}"
    fi
    if [[ -n "${EVENTS_LOG_ABS:-}" ]]; then
        echo "events_log ${EVENTS_LOG_ABS}"
    fi
    if [[ -f "${RECORD_ABS:-}" ]]; then
        echo "experience_log ${RECORD_ABS}"
    fi
    if [[ -f "${METRICS_SUMMARY_ABS:-}" ]]; then
        echo "metrics_summary ${METRICS_SUMMARY_ABS}"
        echo "metrics_episodes $(read_summary_value "${METRICS_SUMMARY_ABS}" episodes)"
        echo "metrics_win_rate $(read_summary_value "${METRICS_SUMMARY_ABS}" win_rate)"
        echo "metrics_average_reward $(read_summary_value "${METRICS_SUMMARY_ABS}" average_reward)"
        echo "metrics_average_distance $(read_summary_value "${METRICS_SUMMARY_ABS}" average_distance)"
        echo "metrics_average_predicted_value $(read_summary_value "${METRICS_SUMMARY_ABS}" average_predicted_value)"
        echo "rolling100_win_rate ${rolling100_win_rate}"
        echo "rolling100_average_reward ${rolling100_reward}"
        echo "rolling100_average_distance ${rolling100_distance}"
        echo "rolling500_win_rate ${rolling500_win_rate}"
        echo "rolling500_average_reward ${rolling500_reward}"
        echo "rolling500_average_distance ${rolling500_distance}"
    fi
    if [[ -n "${latest_checkpoint}" ]]; then
        echo "latest_checkpoint ${latest_checkpoint}"
    fi
    if [[ -n "${CHOSEN_CHECKPOINT_ABS:-}" ]]; then
        echo "chosen_checkpoint ${CHOSEN_CHECKPOINT_ABS}"
    fi
    if [[ -n "${BENCH_REPORT_ABS:-}" ]]; then
        echo "bench_report ${BENCH_REPORT_ABS}"
    fi
    if [[ -n "${PROMOTION_STATUS:-}" ]]; then
        echo "promotion_status ${PROMOTION_STATUS}"
    fi
    if [[ -n "${latest_eval_report}" ]]; then
        echo "latest_eval_report ${latest_eval_report}"
    fi
    if [[ -n "${champion_model}" ]]; then
        echo "champion_model ${champion_model}"
    fi

    print_source_progress_status
}

render_monitor() {
    load_current_run_state

    local now
    local phase_display
    local cycle_display
    local elapsed_display="n/a"
    local remaining_display="Until stopped"
    local cycle_remaining_display="n/a"
    local sync_elapsed_display=""
    local last_sync_display="n/a"
    local avg_sync_seconds="n/a"
    local model_episodes="0"
    local model_updates="0"
    local metrics_episodes="0"
    local metrics_win_rate="n/a"
    local metrics_reward="n/a"
    local metrics_distance="n/a"
    local metrics_value="n/a"
    local metrics_steps="n/a"
    local rolling100_win_rate="n/a"
    local rolling100_reward="n/a"
    local rolling100_distance="n/a"
    local rolling500_win_rate="n/a"
    local rolling500_reward="n/a"
    local rolling500_distance="n/a"
    local running_workers="0"
    local worker_bytes="0"
    local experience_growth_bytes="0"
    local recent_events=""
    local runtime_phase_value=""
    local overall_progress_bar=""
    local overall_progress_percent=""
    local cycle_progress_bar=""
    local cycle_progress_percent=""
    local worker_bar=""
    local worker_percent=""
    local sync_activity_bar=""
    local cycle_start_at=""
    local cycle_total_seconds=0
    local cycle_elapsed_seconds=0
    local total_duration_seconds=0
    local total_duration_display=""
    local champion_model=""

    now="$(date +%s)"
    runtime_phase_value="$(runtime_phase)"
    phase_display="$(phase_label "${runtime_phase_value}")"
    cycle_display="${CURRENT_CYCLE:-0}"
    model_episodes="$(blacklight_model_episodes "${MODEL_ABS:-}")"
    model_updates="$(blacklight_model_updates "${MODEL_ABS:-}")"
    running_workers="$(running_worker_count)"
    worker_bytes="$(worker_experience_bytes)"
    experience_growth_bytes="$(blacklight_current_experience_growth "${worker_bytes}" "${LAST_SYNC_EXPERIENCE_BYTES:-0}")"
    avg_sync_seconds="$(blacklight_read_sync_metric "${EVENTS_LOG_ABS:-/dev/null}" avg)"
    champion_model="$(blacklight_current_champion_model || true)"

    if [[ -f "${METRICS_SUMMARY_ABS:-}" ]]; then
        metrics_episodes="$(read_summary_value "${METRICS_SUMMARY_ABS}" episodes)"
        metrics_win_rate="$(read_summary_value "${METRICS_SUMMARY_ABS}" win_rate)"
        metrics_reward="$(read_summary_value "${METRICS_SUMMARY_ABS}" average_reward)"
        metrics_distance="$(read_summary_value "${METRICS_SUMMARY_ABS}" average_distance)"
        metrics_value="$(read_summary_value "${METRICS_SUMMARY_ABS}" average_predicted_value)"
        metrics_steps="$(read_summary_value "${METRICS_SUMMARY_ABS}" average_steps)"
    fi
    if [[ -f "${METRICS_ABS:-}" ]]; then
        rolling100_win_rate="$(blacklight_read_metric_window "${METRICS_ABS}" 100 2)"
        rolling100_reward="$(blacklight_read_metric_window "${METRICS_ABS}" 100 3)"
        rolling100_distance="$(blacklight_read_metric_window "${METRICS_ABS}" 100 4)"
        rolling500_win_rate="$(blacklight_read_metric_window "${METRICS_ABS}" 500 2)"
        rolling500_reward="$(blacklight_read_metric_window "${METRICS_ABS}" 500 3)"
        rolling500_distance="$(blacklight_read_metric_window "${METRICS_ABS}" 500 4)"
    fi

    if [[ -n "${TRAINING_STARTED_AT:-}" && "${TRAINING_STARTED_AT}" =~ ^[0-9]+$ ]]; then
        elapsed_display="$(blacklight_format_duration $(( now - TRAINING_STARTED_AT )))"
    fi
    if [[ -n "${TRAINING_STARTED_AT:-}" && -n "${TRAINING_END_AT:-}" &&
          "${TRAINING_STARTED_AT}" =~ ^[0-9]+$ && "${TRAINING_END_AT}" =~ ^[0-9]+$ &&
          TRAINING_END_AT > TRAINING_STARTED_AT ]]; then
        total_duration_seconds=$(( TRAINING_END_AT - TRAINING_STARTED_AT ))
        total_duration_display="$(blacklight_format_duration "${total_duration_seconds}")"
        overall_progress_bar="$(blacklight_render_progress_bar $(( now - TRAINING_STARTED_AT )) "${total_duration_seconds}" 28)"
        overall_progress_percent="$(blacklight_progress_percent $(( now - TRAINING_STARTED_AT )) "${total_duration_seconds}")"
    fi
    if [[ "${runtime_phase_value}" == "finished" ]]; then
        remaining_display="00:00:00"
    elif [[ -n "${TRAINING_END_AT:-}" && "${TRAINING_END_AT}" =~ ^[0-9]+$ ]]; then
        remaining_display="$(blacklight_format_duration $(( TRAINING_END_AT - now )))"
    fi
    if [[ -n "${CYCLE_DEADLINE:-}" && "${CYCLE_DEADLINE}" =~ ^[0-9]+$ ]]; then
        cycle_remaining_display="$(blacklight_format_duration $(( CYCLE_DEADLINE - now )))"
    fi
    if [[ -n "${SYNC_STARTED_AT:-}" && "${SYNC_STARTED_AT}" =~ ^[0-9]+$ && "${runtime_phase_value}" == "syncing" ]]; then
        sync_elapsed_display="$(blacklight_format_duration $(( now - SYNC_STARTED_AT )))"
        sync_activity_bar="$(blacklight_render_activity_bar $(( now - SYNC_STARTED_AT )) 22)"
    fi
    if [[ -n "${LAST_SYNC_DURATION:-}" && "${LAST_SYNC_DURATION}" =~ ^[0-9]+$ ]]; then
        last_sync_display="$(blacklight_format_duration "${LAST_SYNC_DURATION}")"
    fi
    if [[ -f "${EVENTS_LOG_ABS:-}" ]]; then
        recent_events="$(tail -n 6 "${EVENTS_LOG_ABS}")"
    fi
    if [[ -n "${PARALLEL_WORKERS:-}" && "${PARALLEL_WORKERS}" =~ ^[0-9]+$ && PARALLEL_WORKERS > 0 ]]; then
        worker_bar="$(blacklight_render_progress_bar "${running_workers}" "${PARALLEL_WORKERS}" 22)"
        worker_percent="$(blacklight_progress_percent "${running_workers}" "${PARALLEL_WORKERS}")"
    fi
    if [[ "${runtime_phase_value}" == "collecting" &&
          -n "${CYCLE_DEADLINE:-}" && "${CYCLE_DEADLINE}" =~ ^[0-9]+$ ]]; then
        cycle_start_at="${LAST_SYNC_FINISHED_AT:-${TRAINING_STARTED_AT:-0}}"
        if [[ "${cycle_start_at}" =~ ^[0-9]+$ ]] && (( CYCLE_DEADLINE > cycle_start_at )); then
            cycle_total_seconds=$(( CYCLE_DEADLINE - cycle_start_at ))
            cycle_elapsed_seconds=$(( now - cycle_start_at ))
            cycle_progress_bar="$(blacklight_render_progress_bar "${cycle_elapsed_seconds}" "${cycle_total_seconds}" 22)"
            cycle_progress_percent="$(blacklight_progress_percent "${cycle_elapsed_seconds}" "${cycle_total_seconds}")"
        fi
    fi

    printf 'Blacklight Monitor\n'
    printf '==================\n\n'

    print_status_line "Run" "${RUN_NAME:-unknown}"
    print_status_line "Generation" "${GENERATION:-unknown}"
    print_status_line "Profile" "${PROFILE:-none}"
    print_status_line "Mode" "${TRAINING_MODE:-unknown}"
    print_status_line "Phase" "${phase_display}"
    print_status_line "Cycle" "${cycle_display}"
    print_status_line "Workers" "${running_workers}/${PARALLEL_WORKERS:-0} active"
    print_status_line "Elapsed" "${elapsed_display}"
    print_status_line "Remaining" "${remaining_display}"
    if [[ -n "${overall_progress_bar}" ]]; then
        print_status_line "Overall" "${overall_progress_bar} ${overall_progress_percent}%% of ${total_duration_display}"
    fi
    if [[ -n "${worker_bar}" ]]; then
        print_status_line "Worker Load" "${worker_bar} ${worker_percent}%%"
    fi
    if [[ "${runtime_phase_value}" == "collecting" ]]; then
        print_status_line "Cycle Remaining" "${cycle_remaining_display}"
        if [[ -n "${cycle_progress_bar}" ]]; then
            print_status_line "Cycle Window" "${cycle_progress_bar} ${cycle_progress_percent}%%"
        fi
    fi
    if [[ -n "${sync_elapsed_display}" ]]; then
        print_status_line "Sync Elapsed" "${sync_elapsed_display}"
        print_status_line "Sync Activity" "${sync_activity_bar}"
    fi
    print_status_line "Last Sync" "${last_sync_display}"
    print_status_line "Avg Sync (s)" "${avg_sync_seconds}"
    print_status_line "Exp Growth" "$(blacklight_human_bytes "${experience_growth_bytes}")"
    if [[ -n "${champion_model}" ]]; then
        print_status_line "Champion" "${champion_model}"
    fi
    printf '\n'

    print_status_line "Model Episodes" "${model_episodes}"
    print_status_line "Model Updates" "${model_updates}"
    print_status_line "Metrics Episodes" "${metrics_episodes:-0}"
    print_status_line "Win Rate" "${metrics_win_rate:-n/a}"
    print_status_line "Avg Reward" "${metrics_reward:-n/a}"
    print_status_line "Avg Distance" "${metrics_distance:-n/a}"
    print_status_line "Avg Value" "${metrics_value:-n/a}"
    print_status_line "Avg Steps" "${metrics_steps:-n/a}"
    print_status_line "Rolling100 Win" "${rolling100_win_rate}"
    print_status_line "Rolling100 Rwd" "${rolling100_reward}"
    print_status_line "Rolling100 Dist" "${rolling100_distance}"
    print_status_line "Rolling500 Win" "${rolling500_win_rate}"
    print_status_line "Rolling500 Rwd" "${rolling500_reward}"
    print_status_line "Rolling500 Dist" "${rolling500_distance}"
    print_status_line "Experience" "$(blacklight_human_bytes "${worker_bytes}")"
    printf '\n'

    print_status_line "Run Dir" "${RUN_DIR_ABS:-unknown}"
    print_status_line "Model File" "${MODEL_ABS:-missing}"
    if [[ -n "${PARENT_MODEL:-}" ]]; then
        print_status_line "Parent Model" "${PARENT_MODEL}"
    fi
    if [[ -n "${CHOSEN_CHECKPOINT_ABS:-}" ]]; then
        print_status_line "Chosen Checkpoint" "${CHOSEN_CHECKPOINT_ABS}"
    fi
    if [[ -n "${BENCH_REPORT_ABS:-}" ]]; then
        print_status_line "Bench Report" "${BENCH_REPORT_ABS}"
    fi
    if [[ -n "${PROMOTION_STATUS:-}" ]]; then
        print_status_line "Promotion" "${PROMOTION_STATUS}"
    fi
    if [[ -n "${TRAINER_LOG_ABS:-}" ]]; then
        print_status_line "Trainer Log" "${TRAINER_LOG_ABS}"
    fi
    if [[ -n "${EVENTS_LOG_ABS:-}" ]]; then
        print_status_line "Events Log" "${EVENTS_LOG_ABS}"
    fi

    if [[ -n "${SOURCE_LIST_ABS:-}" && -n "${STATE_FILE_ABS:-}" ]]; then
        printf '\nSource progress\n'
        printf '%s\n' '---------------'
        while IFS='|' read -r source_path source_episodes source_offset; do
            [[ -n "${source_path}" ]] || continue
            printf '%s | episodes %s | offset %s bytes\n' "${source_path}" "${source_episodes}" "${source_offset}"
        done < <(blacklight_format_source_progress "${STATE_FILE_ABS}" "${SOURCE_LIST_ABS}")
    fi

    if [[ -n "${recent_events}" ]]; then
        printf '\nRecent events\n'
        printf '%s\n' '-------------'
        printf '%s\n' "${recent_events}"
    fi
}

run_monitor() {
    local interval="2"
    local once="0"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --interval)
                interval="$2"
                shift 2
                ;;
            --once)
                once="1"
                shift
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                echo "Unknown monitor option: $1" >&2
                usage >&2
                exit 1
                ;;
        esac
    done

    if ! [[ "${interval}" =~ ^[0-9]+$ ]] || (( interval < 1 )); then
        echo "Monitor interval must be a positive integer." >&2
        exit 1
    fi

    while :; do
        if [[ "${once}" == "0" && -t 1 ]]; then
            clear
        fi
        render_monitor
        if [[ "${once}" == "1" ]]; then
            break
        fi
        sleep "${interval}"
    done
}

run_train() {
    local profile=""
    local duration=""
    local rounds=""
    local checkpoint_every=""
    local resume_model=""
    local generation=""
    local run_name=""
    local bin_path=""
    local fast_mode="0"
    local heavy_mode="0"
    local parallel_workers=""
    local sync_seconds=""
    local resolved_resume_model=""
    local training_mode=""
    local profile_default=""
    local profile_value=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile)
                profile="$2"
                shift 2
                ;;
            --duration)
                duration="$2"
                shift 2
                ;;
            --rounds)
                rounds="$2"
                shift 2
                ;;
            --checkpoint-every)
                checkpoint_every="$2"
                shift 2
                ;;
            --resume)
                resume_model="$2"
                shift 2
                ;;
            --generation)
                generation="$2"
                shift 2
                ;;
            --name)
                run_name="$2"
                shift 2
                ;;
            --bin)
                bin_path="$2"
                shift 2
                ;;
            --fast)
                fast_mode="1"
                shift
                ;;
            --heavy)
                heavy_mode="1"
                shift
                ;;
            --parallel-workers)
                parallel_workers="$2"
                shift 2
                ;;
            --sync-seconds)
                sync_seconds="$2"
                shift 2
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                echo "Unknown train option: $1" >&2
                usage >&2
                exit 1
                ;;
        esac
    done

    profile_default="$(blacklight_train_profile_default)"
    if [[ -z "${profile}" ]]; then
        profile="${profile_default}"
    fi
    if ! blacklight_train_profile_exists "${profile}"; then
        echo "Unknown Blacklight training profile: ${profile}" >&2
        exit 1
    fi

    if [[ "${profile}" != "none" ]]; then
        profile_value="$(blacklight_train_profile_value "${profile}" training_mode)"
        if [[ -z "${training_mode}" && -n "${profile_value}" ]]; then
            training_mode="${profile_value}"
        fi

        [[ -n "${duration}" ]] || duration="$(blacklight_train_profile_value "${profile}" duration_seconds)"
        [[ -n "${checkpoint_every}" ]] || checkpoint_every="$(blacklight_train_profile_value "${profile}" checkpoint_every)"
        [[ -n "${parallel_workers}" ]] || parallel_workers="$(blacklight_train_profile_value "${profile}" parallel_workers)"
        [[ -n "${sync_seconds}" ]] || sync_seconds="$(blacklight_train_profile_value "${profile}" sync_seconds)"

        export ARMAGETRON_SELFPLAY_PROFILE="${profile}"
        profile_value="$(blacklight_train_profile_value "${profile}" bot_count)"
        [[ -n "${profile_value}" ]] && export ARMAGETRON_SELFPLAY_BOT_COUNT="${profile_value}"
        profile_value="$(blacklight_train_profile_value "${profile}" exploration)"
        [[ -n "${profile_value}" ]] && export ARMAGETRON_SELFPLAY_EXPLORATION="${profile_value}"
        profile_value="$(blacklight_train_profile_value "${profile}" train_epochs)"
        [[ -n "${profile_value}" ]] && export ARMAGETRON_SELFPLAY_TRAIN_EPOCHS="${profile_value}"
        profile_value="$(blacklight_train_profile_value "${profile}" record_stride)"
        [[ -n "${profile_value}" ]] && export ARMAGETRON_SELFPLAY_RECORD_STRIDE="${profile_value}"
        [[ -n "${parallel_workers}" ]] && export ARMAGETRON_SELFPLAY_PARALLEL_RECORD_STRIDE="$(blacklight_train_profile_value "${profile}" record_stride)"
        profile_value="$(blacklight_train_profile_value "${profile}" save_every)"
        [[ -n "${profile_value}" && -z "${ARMAGETRON_SELFPLAY_SAVE_EVERY:-}" ]] && export ARMAGETRON_SELFPLAY_SAVE_EVERY="${profile_value}"
        profile_value="$(blacklight_train_profile_value "${profile}" policy_pool_size)"
        [[ -n "${profile_value}" ]] && export ARMAGETRON_SELFPLAY_POLICY_POOL_SIZE="${profile_value}"
        profile_value="$(blacklight_train_profile_value "${profile}" policy_snapshot_every)"
        [[ -n "${profile_value}" ]] && export ARMAGETRON_SELFPLAY_POLICY_SNAPSHOT_EVERY="${profile_value}"
        profile_value="$(blacklight_train_profile_value "${profile}" policy_snapshot_warmup)"
        [[ -n "${profile_value}" ]] && export ARMAGETRON_SELFPLAY_POLICY_SNAPSHOT_WARMUP="${profile_value}"
        profile_value="$(blacklight_train_profile_value "${profile}" policy_historic_prob)"
        [[ -n "${profile_value}" ]] && export ARMAGETRON_SELFPLAY_POLICY_HISTORIC_PROB="${profile_value}"
        profile_value="$(blacklight_train_profile_value "${profile}" min_players)"
        [[ -n "${profile_value}" ]] && export ARMAGETRON_SELFPLAY_MIN_PLAYERS="${profile_value}"
        profile_value="$(blacklight_train_profile_value "${profile}" teams_min)"
        [[ -n "${profile_value}" ]] && export ARMAGETRON_SELFPLAY_TEAMS_MIN="${profile_value}"
        profile_value="$(blacklight_train_profile_value "${profile}" teams_max)"
        [[ -n "${profile_value}" ]] && export ARMAGETRON_SELFPLAY_TEAMS_MAX="${profile_value}"
    else
        export ARMAGETRON_SELFPLAY_PROFILE="none"
    fi

    if [[ -z "${training_mode}" ]]; then
        if [[ -n "${parallel_workers}" || -n "${sync_seconds}" ]]; then
            training_mode="parallel"
        else
            training_mode="selfplay"
        fi
    fi

    if [[ -n "${resume_model}" ]]; then
        if ! resolved_resume_model="$(blacklight_resolve_model_input "${resume_model}" 1)"; then
            echo "Could not resolve resume model: ${resume_model}" >&2
            exit 1
        fi
        export ARMAGETRON_SELFPLAY_INITIAL_MODEL="${resolved_resume_model}"
        export ARMAGETRON_SELFPLAY_PARENT_MODEL="${resolved_resume_model}"
    elif [[ "${profile}" == "classic_hardening" || "${profile}" == "mixed_league" ]]; then
        echo "Profile ${profile} requires --resume MODEL_PATH_OR_RUN." >&2
        exit 1
    fi

    if [[ -n "${duration}" ]]; then
        export ARMAGETRON_SELFPLAY_DURATION_SECONDS="${duration}"
    fi
    if [[ -n "${rounds}" ]]; then
        export ARMAGETRON_SELFPLAY_LIMIT_ROUNDS="${rounds}"
    fi
    if [[ -n "${checkpoint_every}" ]]; then
        export ARMAGETRON_SELFPLAY_CHECKPOINT_EVERY="${checkpoint_every}"
    fi
    if [[ -n "${generation}" ]]; then
        export ARMAGETRON_SELFPLAY_GENERATION="${generation}"
    fi
    if [[ -n "${run_name}" ]]; then
        export ARMAGETRON_SELFPLAY_RUN_NAME="${run_name}"
    fi
    if [[ -n "${bin_path}" ]]; then
        export ARMAGETRON_SELFPLAY_BIN="${bin_path}"
    fi
    if [[ "${fast_mode}" == "1" ]]; then
        export ARMAGETRON_SELFPLAY_FAST_MODE="1"
    fi
    if [[ "${heavy_mode}" == "1" ]]; then
        export ARMAGETRON_SELFPLAY_HEAVY_MODE="1"
    fi
    if [[ -n "${parallel_workers}" ]]; then
        export ARMAGETRON_SELFPLAY_PARALLEL_WORKERS="${parallel_workers}"
    fi
    if [[ -n "${sync_seconds}" ]]; then
        export ARMAGETRON_SELFPLAY_SYNC_SECONDS="${sync_seconds}"
    fi

    if [[ "${training_mode}" == "parallel" ]]; then
        if [[ "${fast_mode}" == "1" ]]; then
            echo "Parallel Blacklight training needs worker experience logs, so --fast cannot be combined with parallel training." >&2
            exit 1
        fi
        exec "${SCRIPT_DIR}/train_neural_ai_parallel.sh"
    fi

    exec "${SCRIPT_DIR}/train_neural_ai_selfplay.sh"
}

run_smoke() {
    local duration=""
    local rounds=""
    local bin_path=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --duration)
                duration="$2"
                shift 2
                ;;
            --rounds)
                rounds="$2"
                shift 2
                ;;
            --bin)
                bin_path="$2"
                shift 2
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                echo "Unknown smoke option: $1" >&2
                usage >&2
                exit 1
                ;;
        esac
    done

    if [[ -n "${duration}" ]]; then
        export ARMAGETRON_SELFPLAY_DURATION_SECONDS="${duration}"
    fi
    if [[ -n "${rounds}" ]]; then
        export ARMAGETRON_SELFPLAY_LIMIT_ROUNDS="${rounds}"
    fi
    if [[ -n "${bin_path}" ]]; then
        export ARMAGETRON_SELFPLAY_BIN="${bin_path}"
    fi

    exec "${SCRIPT_DIR}/smoke_test_blacklight_training.sh"
}

run_eval() {
    local duration=""
    local rounds=""
    local candidate=""
    local reference=""
    local bin_path=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --duration)
                duration="$2"
                shift 2
                ;;
            --rounds)
                rounds="$2"
                shift 2
                ;;
            --candidate)
                candidate="$2"
                shift 2
                ;;
            --reference)
                reference="$2"
                shift 2
                ;;
            --bin)
                bin_path="$2"
                shift 2
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                echo "Unknown eval option: $1" >&2
                usage >&2
                exit 1
                ;;
        esac
    done

    if [[ -n "${duration}" ]]; then
        export ARMAGETRON_EVAL_DURATION_SECONDS="${duration}"
    fi
    if [[ -n "${rounds}" ]]; then
        export ARMAGETRON_EVAL_LIMIT_ROUNDS="${rounds}"
    fi
    if [[ -n "${bin_path}" ]]; then
        export ARMAGETRON_SELFPLAY_BIN="${bin_path}"
    fi

    if [[ -n "${candidate}" && -n "${reference}" ]]; then
        exec "${SCRIPT_DIR}/evaluate_blacklight.sh" "${candidate}" "${reference}"
    elif [[ -n "${candidate}" ]]; then
        exec "${SCRIPT_DIR}/evaluate_blacklight.sh" "${candidate}"
    else
        exec "${SCRIPT_DIR}/evaluate_blacklight.sh"
    fi
}

run_bench() {
    local suite=""
    local candidate=""
    local reference=""
    local bin_path=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --suite)
                suite="$2"
                shift 2
                ;;
            --candidate)
                candidate="$2"
                shift 2
                ;;
            --reference)
                reference="$2"
                shift 2
                ;;
            --bin)
                bin_path="$2"
                shift 2
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                echo "Unknown bench option: $1" >&2
                usage >&2
                exit 1
                ;;
        esac
    done

    [[ -n "${suite}" ]] || { echo "bench requires --suite." >&2; exit 1; }
    [[ -n "${candidate}" ]] || { echo "bench requires --candidate." >&2; exit 1; }
    if [[ -n "${bin_path}" ]]; then
        export ARMAGETRON_SELFPLAY_BIN="${bin_path}"
    fi

    if [[ -n "${reference}" ]]; then
        exec "${SCRIPT_DIR}/benchmark_blacklight.sh" --suite "${suite}" --candidate "${candidate}" --reference "${reference}"
    fi
    exec "${SCRIPT_DIR}/benchmark_blacklight.sh" --suite "${suite}" --candidate "${candidate}"
}

run_promote() {
    local candidate=""
    local bin_path=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --candidate)
                candidate="$2"
                shift 2
                ;;
            --bin)
                bin_path="$2"
                shift 2
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                echo "Unknown promote option: $1" >&2
                usage >&2
                exit 1
                ;;
        esac
    done

    [[ -n "${candidate}" ]] || { echo "promote requires --candidate." >&2; exit 1; }
    if [[ -n "${bin_path}" ]]; then
        export ARMAGETRON_SELFPLAY_BIN="${bin_path}"
    fi

    exec "${SCRIPT_DIR}/promote_blacklight.sh" --candidate "${candidate}"
}

run_sweep() {
    local profile_chain="classic"
    local bin_path=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile-chain)
                profile_chain="$2"
                shift 2
                ;;
            --bin)
                bin_path="$2"
                shift 2
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                echo "Unknown sweep option: $1" >&2
                usage >&2
                exit 1
                ;;
        esac
    done

    if [[ -n "${bin_path}" ]]; then
        export ARMAGETRON_SELFPLAY_BIN="${bin_path}"
    fi

    exec "${SCRIPT_DIR}/sweep_blacklight.sh" --profile-chain "${profile_chain}"
}

COMMAND="${1:-train}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${COMMAND}" in
    train)
        run_train "$@"
        ;;
    smoke)
        run_smoke "$@"
        ;;
    eval)
        run_eval "$@"
        ;;
    bench)
        run_bench "$@"
        ;;
    promote)
        run_promote "$@"
        ;;
    sweep)
        run_sweep "$@"
        ;;
    monitor)
        run_monitor "$@"
        ;;
    status)
        if [[ $# -gt 0 ]]; then
            echo "status does not take options." >&2
            usage >&2
            exit 1
        fi
        show_status
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown Blacklight command: ${COMMAND}" >&2
        usage >&2
        exit 1
        ;;
esac
