#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/blacklight_lib.sh"
REPO_ROOT="${BLACKLIGHT_REPO_ROOT}"

usage() {
    cat <<'EOF'
Blacklight helper

Usage:
  ./scripts/blacklight.sh collect [options]
  ./scripts/blacklight.sh train [options]
  ./scripts/blacklight.sh smoke [options]
  ./scripts/blacklight.sh eval [options]
  ./scripts/blacklight.sh bench [options]
  ./scripts/blacklight.sh promote [options]
  ./scripts/blacklight.sh current [show|set|distill] [options]
  ./scripts/blacklight.sh vast [setup|pack|apply] [options]
  ./scripts/blacklight.sh champion [show|init] [options]
  ./scripts/blacklight.sh gpu-setup
  ./scripts/blacklight.sh prune [options]
  ./scripts/blacklight.sh sweep [options]
  ./scripts/blacklight.sh pipeline
  ./scripts/blacklight.sh dashboard [options]
  ./scripts/blacklight.sh status
  ./scripts/blacklight.sh monitor [options]
  ./scripts/blacklight.sh help

If no command is given, "train" is used.

Collect options:
  Generates fresh teacher gameplay data from the built-in bots.
  --profile NAME
  --duration SECONDS
  --rounds COUNT
  --resume MODEL_PATH_OR_RUN
  --fresh
  --generation NAME
  --name RUN_NAME
  --bin PATH
  --fast
  --heavy
  --parallel-workers COUNT
  --sync-seconds SECONDS

Train options:
  Trains the current CNN from collected teacher data on GPU/MPS by default.
  --data SOURCE_LIST_OR_RUN
  --profile NAME
  --resume MODEL_PATH_OR_RUN
  --fresh
  --gpu
  --cpu
  --in-place
  --device auto|mps|cuda|cpu
  --batch-size COUNT
  --epochs COUNT
  --lr RATE
  --max-examples COUNT
  --checkpoint-every COUNT
  --duration SECONDS
  --rounds COUNT
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
  --sessions COUNT
  --duration SECONDS
  --rounds COUNT
  --bin PATH

Promote options:
  --candidate MODEL_PATH_OR_RUN
  --bin PATH

Current best options:
  show
  set
  --model MODEL_PATH_OR_RUN
  distill
  --from MODEL_PATH_OR_RUN
  --alpha RATE
  --name RUN_NAME

Vast.ai options:
  setup
  pack
  apply MODEL_PATH
  See ./scripts/blacklight_vast.sh help for packaging options.

Champion options:
  show
  init
  --model MODEL_PATH_OR_RUN
  --replace

GPU setup:
  Creates .venv-blacklight-gpu and installs PyTorch for CPU+GPU teacher replay.

Prune options:
  --dry-run
  --apply
  --keep-checkpoints COUNT
  --skip-model-copies

Sweep options:
  --profile-chain classic
  --bin PATH

Monitor options:
  --interval SECONDS
  --once
  --classic
  --no-color

Dashboard options:
  --host ADDRESS
  --port PORT
  --run RUN_NAME
  --no-open

Pipeline:
  Prints the canonical Blacklight training workflow stored in
  /Users/romilbijarnia/Desktop/Armagetron/docs/BLACKLIGHT_TRAINING_PIPELINE.md
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
            printf 'Collecting training data\n'
            ;;
        syncing)
            printf 'Training model\n'
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

    if [[ -n "${WORKERS_RUNNING:-}" ]]; then
        printf '%s\n' "${WORKERS_RUNNING}"
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

monitor_supports_color() {
    if [[ "${BLACKLIGHT_MONITOR_FORCE_COLOR:-0}" == "1" ]]; then
        return 0
    fi
    if [[ -n "${NO_COLOR:-}" ]]; then
        return 1
    fi

    [[ "${BLACKLIGHT_MONITOR_TTY_COLOR:-0}" == "1" || -t 1 ]]
}

monitor_supports_repaint() {
    if [[ "${BLACKLIGHT_MONITOR_PLAIN:-0}" == "1" ]]; then
        return 1
    fi

    if [[ -t 1 ]]; then
        return 0
    fi

    return 1
}

monitor_width() {
    local width="${COLUMNS:-}"

    if [[ -t 1 ]] && command -v tput >/dev/null 2>&1; then
        width="$(tput cols 2>/dev/null || printf '%s' "${width:-}")"
    fi

    if [[ -z "${width}" || ! "${width}" =~ ^[0-9]+$ ]] || (( width < 48 )); then
        width=80
    elif (( width > 140 )); then
        width=140
    fi

    if (( width > 48 )); then
        width=$(( width - 1 ))
    fi

    printf '%s\n' "${width}"
}

monitor_ansi() {
    local code="$1"
    if monitor_supports_color; then
        printf '\033[%sm' "${code}"
    fi
}

monitor_colorize() {
    local code="$1"
    local text="$2"

    if monitor_supports_color; then
        printf '%s%s%s' "$(monitor_ansi "${code}")" "${text}" "$(monitor_ansi '0')"
    else
        printf '%s' "${text}"
    fi
}

monitor_unicode() {
    [[ "${BLACKLIGHT_MONITOR_ASCII:-0}" != "1" ]]
}

monitor_repeat() {
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

monitor_rail() {
    if monitor_unicode; then
        printf '│'
    else
        printf '|'
    fi
}

monitor_heading() {
    local text="$1"
    local width
    local inner_width
    local content
    local padding
    local horizontal='═'
    local top_left='╔'
    local top_right='╗'
    local bottom_left='╚'
    local bottom_right='╝'
    local vertical='║'

    if ! monitor_unicode; then
        horizontal='='
        top_left='+'
        top_right='+'
        bottom_left='+'
        bottom_right='+'
        vertical='|'
    fi

    width="$(monitor_width)"
    inner_width=$(( width - 2 ))
    content=" BLACKLIGHT TRAINING CONSOLE  ${text#Blacklight Ops  }"
    content="$(monitor_trim_middle "${content}" "${inner_width}")"
    padding=$(( inner_width - ${#content} ))
    if (( padding < 0 )); then
        padding=0
    fi

    printf '%s%s%s\n' \
        "$(monitor_colorize '1;36' "${top_left}")" \
        "$(monitor_colorize '1;36' "$(monitor_repeat "${horizontal}" "${inner_width}")")" \
        "$(monitor_colorize '1;36' "${top_right}")"
    printf '%s%s%s%s\n' \
        "$(monitor_colorize '1;36' "${vertical}")" \
        "$(monitor_colorize '1;37' "${content}")" \
        "$(monitor_repeat ' ' "${padding}")" \
        "$(monitor_colorize '1;36' "${vertical}")"
    printf '%s%s%s\n' \
        "$(monitor_colorize '1;36' "${bottom_left}")" \
        "$(monitor_colorize '1;36' "$(monitor_repeat "${horizontal}" "${inner_width}")")" \
        "$(monitor_colorize '1;36' "${bottom_right}")"
}

monitor_section() {
    local text="$1"
    local width
    local prefix
    local fill_count
    local horizontal='─'
    local corner='╭'

    if ! monitor_unicode; then
        horizontal='-'
        corner='+'
    fi

    width="$(monitor_width)"
    prefix="${corner}${horizontal} ${text} "
    fill_count=$(( width - ${#prefix} ))
    if (( fill_count < 0 )); then
        fill_count=0
    fi

    printf '\n%s%s\n' \
        "$(monitor_colorize '1;34' "${prefix}")" \
        "$(monitor_colorize '2;34' "$(monitor_repeat "${horizontal}" "${fill_count}")")"
}

monitor_badge() {
    local text="$1"
    local color_code="$2"

    if monitor_supports_color; then
        if monitor_unicode; then
            printf '%s● %s%s' "$(monitor_ansi "1;${color_code}")" "${text}" "$(monitor_ansi '0')"
        else
            printf '%s[%s]%s' "$(monitor_ansi "1;${color_code}")" "${text}" "$(monitor_ansi '0')"
        fi
    else
        if monitor_unicode; then
            printf '● %s' "${text}"
        else
            printf '[%s]' "${text}"
        fi
    fi
}

monitor_phase_badge() {
    case "${1:-}" in
        initializing) monitor_badge "INITIALIZING" "35" ;;
        collecting) monitor_badge "COLLECTING" "33" ;;
        syncing) monitor_badge "SYNCING" "36" ;;
        stopping) monitor_badge "STOPPING" "31" ;;
        finished) monitor_badge "FINISHED" "32" ;;
        *) monitor_badge "UNKNOWN" "37" ;;
    esac
}

monitor_rule() {
    local fill_char="${1:-─}"
    local width

    if ! monitor_unicode && [[ "${fill_char}" != "=" ]]; then
        fill_char='-'
    fi

    width="$(monitor_width)"

    printf '%s\n' "$(monitor_repeat "${fill_char}" "${width}")"
}

monitor_progress_bar() {
    local current="${1:-0}"
    local total="${2:-0}"
    local width="${3:-24}"
    local color_code="${4:-36}"
    local filled=0
    local index
    local left='▕'
    local right='▏'
    local fill='█'
    local empty='░'

    if ! monitor_unicode; then
        blacklight_render_progress_bar "${current}" "${total}" "${width}" "#" "."
        return 0
    fi

    if [[ -z "${width}" || ! "${width}" =~ ^[0-9]+$ ]] || (( width < 1 )); then
        width=24
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

    printf '%s' "$(monitor_colorize '2;37' "${left}")"
    for (( index = 0; index < width; ++index )); do
        if (( index < filled )); then
            printf '%s' "$(monitor_colorize "1;${color_code}" "${fill}")"
        else
            printf '%s' "$(monitor_colorize '2;37' "${empty}")"
        fi
    done
    printf '%s' "$(monitor_colorize '2;37' "${right}")"
}

monitor_activity_bar() {
    local tick="${1:-0}"
    local width="${2:-24}"
    local color_code="${3:-36}"
    local index
    local position
    local left='▕'
    local right='▏'
    local track='─'
    local marker='◆'

    if ! monitor_unicode; then
        blacklight_render_activity_bar "${tick}" "${width}"
        return 0
    fi

    if [[ -z "${tick}" || ! "${tick}" =~ ^-?[0-9]+$ ]]; then
        tick=0
    fi
    if [[ -z "${width}" || ! "${width}" =~ ^[0-9]+$ ]] || (( width < 1 )); then
        width=24
    fi
    if (( tick < 0 )); then
        tick=$(( -tick ))
    fi
    position=$(( tick % width ))

    printf '%s' "$(monitor_colorize '2;37' "${left}")"
    for (( index = 0; index < width; ++index )); do
        if (( index == position )); then
            printf '%s' "$(monitor_colorize "1;${color_code}" "${marker}")"
        else
            printf '%s' "$(monitor_colorize '2;37' "${track}")"
        fi
    done
    printf '%s' "$(monitor_colorize '2;37' "${right}")"
}

monitor_numeric_percent() {
    local value="${1:-}"

    if [[ "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        awk -v value="${value}" 'BEGIN {
            if (value < 0) value = 0;
            if (value > 100) value = 100;
            printf "%.0f", value;
        }'
    else
        printf 'n/a'
    fi
}

monitor_load_color() {
    local value="${1:-}"

    if ! [[ "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        printf '37'
        return 0
    fi

    awk -v value="${value}" 'BEGIN {
        if (value >= 85) print 31;
        else if (value >= 65) print 33;
        else print 32;
    }'
}

monitor_system_cpu_percent() {
    top -l 1 -n 0 2>/dev/null | awk '
        /CPU usage/ {
            for (i = 1; i <= NF; ++i) {
                if ($i == "idle") {
                    idle = $(i - 1);
                    gsub("%", "", idle);
                    if (idle ~ /^[0-9.]+$/) {
                        value = 100 - idle;
                        if (value < 0) value = 0;
                        if (value > 100) value = 100;
                        printf "%.0f\n", value;
                    }
                    exit;
                }
            }
        }
    '
}

monitor_gpu_percent() {
    ioreg -r -d 1 -w 0 -c AGXAccelerator 2>/dev/null \
        | sed -n 's/.*"Device Utilization %"=\([0-9][0-9.]*\).*/\1/p' \
        | head -n 1
}

monitor_trainer_cpu_percent() {
    local trainer_pid="${1:-0}"
    local cpu=""

    if [[ "${trainer_pid}" =~ ^[0-9]+$ ]] && (( trainer_pid > 0 )); then
        cpu="$(ps -axo ppid=,%cpu= 2>/dev/null | awk -v parent="${trainer_pid}" '
            $1 == parent { sum += $2 }
            END {
                if (sum > 0) printf "%.0f\n", sum;
            }
        ')"
        if [[ -z "${cpu}" ]]; then
            cpu="$(ps -p "${trainer_pid}" -o %cpu= 2>/dev/null | awk '{ sum += $1 } END { if (sum > 0) printf "%.0f\n", sum }')"
        fi
    fi

    if [[ -n "${cpu}" ]]; then
        printf '%s\n' "${cpu}"
    else
        printf '0\n'
    fi
}

monitor_load_block() {
    local value="${1:-}"
    local blocks=(▁ ▂ ▃ ▄ ▅ ▆ ▇ █)
    local ascii_blocks=(. . - - = = '#' '@')
    local bucket=0

    if ! [[ "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        if monitor_unicode; then
            printf '·'
        else
            printf '.'
        fi
        return 0
    fi

    bucket="$(awk -v value="${value}" 'BEGIN {
        if (value < 0) value = 0;
        if (value > 100) value = 100;
        printf "%d", int(value / 12.5);
    }')"
    if (( bucket > 7 )); then
        bucket=7
    fi

    if monitor_unicode; then
        printf '%s' "${blocks[${bucket}]}"
    else
        printf '%s' "${ascii_blocks[${bucket}]}"
    fi
}

monitor_append_load_sample() {
    local history_path="$1"
    local sample_time="$2"
    local cpu_value="$3"
    local gpu_value="$4"
    local trainer_cpu_value="$5"
    local history_dir
    local tmp_path

    [[ -n "${history_path}" ]] || return 0
    history_dir="$(dirname "${history_path}")"
    mkdir -p "${history_dir}" 2>/dev/null || return 0

    [[ "${cpu_value}" =~ ^[0-9]+([.][0-9]+)?$ ]] || cpu_value="-1"
    [[ "${gpu_value}" =~ ^[0-9]+([.][0-9]+)?$ ]] || gpu_value="-1"
    [[ "${trainer_cpu_value}" =~ ^[0-9]+([.][0-9]+)?$ ]] || trainer_cpu_value="-1"

    printf '%s\t%s\t%s\t%s\n' "${sample_time}" "${cpu_value}" "${gpu_value}" "${trainer_cpu_value}" >> "${history_path}"
    tmp_path="${history_path}.tmp"
    tail -n 80 "${history_path}" > "${tmp_path}" 2>/dev/null && mv "${tmp_path}" "${history_path}" 2>/dev/null || rm -f "${tmp_path}"
}

monitor_load_sparkline() {
    local history_path="$1"
    local column="$2"
    local width="${3:-48}"
    local value
    local count=0
    local pad_count=0
    local sparkline=""

    if [[ -z "${width}" || ! "${width}" =~ ^[0-9]+$ ]] || (( width < 8 )); then
        width=48
    fi

    if [[ -f "${history_path}" ]]; then
        while IFS=$'\t' read -r _sample_time sample_cpu sample_gpu sample_trainer_cpu; do
            case "${column}" in
                cpu) value="${sample_cpu}" ;;
                gpu) value="${sample_gpu}" ;;
                trainer) value="${sample_trainer_cpu}" ;;
                *) value="-1" ;;
            esac
            if [[ "${value}" == "-1" ]]; then
                value=""
            fi
            sparkline="${sparkline}$(monitor_load_block "${value}")"
            count=$(( count + 1 ))
        done < <(tail -n "${width}" "${history_path}" 2>/dev/null)
    fi

    pad_count=$(( width - count ))
    while (( pad_count > 0 )); do
        sparkline="$(monitor_load_block "")${sparkline}"
        pad_count=$(( pad_count - 1 ))
    done

    printf '%s' "${sparkline}"
}

monitor_load_row() {
    local label="$1"
    local value="$2"
    local graph="$3"
    local note="$4"
    local percent
    local color_code

    percent="$(monitor_numeric_percent "${value}")"
    color_code="$(monitor_load_color "${value}")"
    if [[ "${percent}" == "n/a" ]]; then
        printf '%s %-14s %-5s %s %s\n' \
            "$(monitor_colorize '2;37' "$(monitor_rail)")" \
            "${label}" \
            "n/a" \
            "$(monitor_colorize '2;37' "${graph}")" \
            "${note}"
    else
        printf '%s %-14s %s %s %s\n' \
            "$(monitor_colorize '2;37' "$(monitor_rail)")" \
            "${label}" \
            "$(monitor_colorize "1;${color_code}" "$(printf '%3s%%' "${percent}")")" \
            "$(monitor_colorize "${color_code}" "${graph}")" \
            "${note}"
    fi
}

monitor_trim_middle() {
    local text="$1"
    local max_len="${2:-0}"
    local prefix_len=0
    local suffix_len=0

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

    prefix_len=$(( (max_len - 3) / 2 ))
    suffix_len=$(( max_len - 3 - prefix_len ))

    if (( suffix_len > 0 )); then
        printf '%s...%s' "${text:0:prefix_len}" "${text: -suffix_len}"
    else
        printf '%s...' "${text:0:prefix_len}"
    fi
}

monitor_fit_value() {
    local value="$1"
    local reserved_width="${2:-18}"
    local width
    local max_len

    width="$(monitor_width)"
    max_len=$(( width - reserved_width ))
    if (( max_len < 18 )); then
        max_len=18
    fi

    monitor_trim_middle "${value}" "${max_len}"
}

monitor_display_path() {
    local value="$1"

    if [[ -n "${REPO_ROOT:-}" && "${value}" == "${REPO_ROOT}/"* ]]; then
        printf '%s\n' "${value#"${REPO_ROOT}/"}"
        return 0
    fi

    if [[ -n "${HOME:-}" && "${value}" == "${HOME}/"* ]]; then
        printf '~/%s\n' "${value#"${HOME}/"}"
        return 0
    fi

    printf '%s\n' "${value}"
}

monitor_compact_row() {
    local width
    local first_value=""
    local second_value=""
    local third_value=""
    local rail=""

    width="$(monitor_width)"
    rail="$(monitor_colorize '2;37' "$(monitor_rail)")"
    first_value="$(monitor_trim_middle "$2" 16)"
    second_value="$(monitor_trim_middle "$4" 18)"
    third_value="$(monitor_fit_value "$6" 44)"

    if (( width < 96 )); then
        monitor_detail_row "$1" "$(monitor_fit_value "$2" 18)"
        monitor_detail_row "$3" "$(monitor_fit_value "$4" 18)"
        monitor_detail_row "$5" "$(monitor_fit_value "$6" 18)"
        return 0
    fi

    if (( width < 144 )); then
        printf '%s %-12s %-16s %-12s %s\n' "${rail}" "$1" "${first_value}" "$3" "$(monitor_fit_value "$4" 34)"
        monitor_detail_row "$5" "$(monitor_fit_value "$6" 18)"
        return 0
    fi

    printf '%s %-12s %-16s %-12s %-18s %-12s %s\n' "${rail}" "$1" "${first_value}" "$3" "${second_value}" "$5" "${third_value}"
}

monitor_detail_row() {
    printf '%s %-14s %s\n' "$(monitor_colorize '2;37' "$(monitor_rail)")" "$1" "$2"
}

monitor_path_row() {
    local label="$1"
    local value="$2"
    monitor_detail_row "${label}" "$(monitor_fit_value "$(monitor_display_path "${value}")" 18)"
}

monitor_source_label() {
    local source_path="$1"
    local source_dir

    source_dir="$(dirname "${source_path}")"
    if [[ "${source_dir}" == */workers/* ]]; then
        basename "${source_dir}"
        return 0
    fi

    basename "${source_path}"
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
    local rolling100_policy_loss="n/a"
    local rolling100_value_loss="n/a"
    local rolling100_entropy="n/a"
    local rolling500_win_rate="n/a"
    local rolling500_reward="n/a"
    local rolling500_distance="n/a"
    local metrics_episodes="0"
    local metrics_win_rate="n/a"
    local metrics_reward="n/a"
    local metrics_distance="n/a"
    local metrics_value="n/a"
    local metrics_policy_loss="n/a"
    local metrics_value_loss="n/a"
    local metrics_entropy="n/a"
    local metrics_steps_total="n/a"
    local metrics_learned_episodes="n/a"
    local metrics_steps="n/a"
    local episode_rate="n/a"
    local step_rate="n/a"
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
        rolling100_policy_loss="$(blacklight_read_metric_window "${METRICS_ABS}" 100 14 7 1)"
        rolling100_value_loss="$(blacklight_read_metric_window "${METRICS_ABS}" 100 15 7 1)"
        rolling100_entropy="$(blacklight_read_metric_window "${METRICS_ABS}" 100 16 7 1)"
        rolling500_win_rate="$(blacklight_read_metric_window "${METRICS_ABS}" 500 2)"
        rolling500_reward="$(blacklight_read_metric_window "${METRICS_ABS}" 500 3)"
        rolling500_distance="$(blacklight_read_metric_window "${METRICS_ABS}" 500 4)"
    fi

    champion_model="$(blacklight_current_champion_model || true)"

    now="$(date +%s)"
    if [[ -n "${TRAINING_STARTED_AT:-}" && "${TRAINING_STARTED_AT}" =~ ^[0-9]+$ ]] && (( TRAINING_STARTED_AT > 0 )); then
        elapsed_display="$(blacklight_format_duration $(( now - TRAINING_STARTED_AT )))"
        if [[ -f "${METRICS_SUMMARY_ABS:-}" ]]; then
            metrics_steps_total="$(read_summary_value "${METRICS_SUMMARY_ABS}" steps_total)"
            metrics_learned_episodes="$(read_summary_value "${METRICS_SUMMARY_ABS}" learned_episodes)"
            metrics_policy_loss="$(read_summary_value "${METRICS_SUMMARY_ABS}" average_policy_loss)"
            metrics_value_loss="$(read_summary_value "${METRICS_SUMMARY_ABS}" average_value_loss)"
            metrics_entropy="$(read_summary_value "${METRICS_SUMMARY_ABS}" average_entropy)"
            episode_rate="$(blacklight_format_rate "$(read_summary_value "${METRICS_SUMMARY_ABS}" episodes)" "$(( now - TRAINING_STARTED_AT ))" "/s" 3)"
            step_rate="$(blacklight_format_rate "${metrics_steps_total}" "$(( now - TRAINING_STARTED_AT ))" "/s" 3)"
        fi
    fi
    if [[ "${metrics_episodes:-0}" == "0" ]]; then
        metrics_win_rate="n/a"
        metrics_reward="n/a"
        metrics_distance="n/a"
        metrics_value="n/a"
        metrics_steps="n/a"
        episode_rate="n/a"
    fi
    if [[ "${TRAINING_MODE:-}" == "teacher" ]]; then
        metrics_win_rate="n/a"
        metrics_reward="n/a"
        metrics_distance="n/a"
        metrics_value="n/a"
        rolling100_win_rate="n/a"
        rolling100_reward="n/a"
        rolling100_distance="n/a"
        rolling500_win_rate="n/a"
        rolling500_reward="n/a"
        rolling500_distance="n/a"
    fi
    if [[ "${metrics_steps_total:-0}" == "0" ]]; then
        step_rate="n/a"
    fi
    if [[ "${metrics_learned_episodes:-0}" == "0" ]]; then
        metrics_policy_loss="n/a"
        metrics_value_loss="n/a"
        metrics_entropy="n/a"
    fi
    if [[ "${runtime_phase_value}" == "finished" ]]; then
        remaining_display="00:00:00"
    elif [[ -n "${TRAINING_END_AT:-}" && "${TRAINING_END_AT}" =~ ^[0-9]+$ ]] && (( TRAINING_END_AT > 0 )); then
        local remaining_seconds
        remaining_seconds=$(( TRAINING_END_AT - now ))
        if (( remaining_seconds < 0 )); then
            remaining_seconds=0
        fi
        remaining_display="$(blacklight_format_duration "${remaining_seconds}")"
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
        echo "metrics_episodes ${metrics_episodes:-0}"
        echo "metrics_win_rate ${metrics_win_rate:-n/a}"
        echo "metrics_average_reward ${metrics_reward:-n/a}"
        echo "metrics_average_distance ${metrics_distance:-n/a}"
        echo "metrics_average_predicted_value ${metrics_value:-n/a}"
        echo "metrics_average_policy_loss ${metrics_policy_loss:-n/a}"
        echo "metrics_average_value_loss ${metrics_value_loss:-n/a}"
        echo "metrics_average_entropy ${metrics_entropy:-n/a}"
        echo "metrics_steps_total ${metrics_steps_total:-n/a}"
        echo "metrics_learned_episodes ${metrics_learned_episodes:-n/a}"
        echo "episode_rate ${episode_rate:-n/a}"
        echo "step_rate ${step_rate:-n/a}"
        echo "rolling100_win_rate ${rolling100_win_rate}"
        echo "rolling100_average_reward ${rolling100_reward}"
        echo "rolling100_average_distance ${rolling100_distance}"
        echo "rolling100_policy_loss ${rolling100_policy_loss}"
        echo "rolling100_value_loss ${rolling100_value_loss}"
        echo "rolling100_entropy ${rolling100_entropy}"
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
    local phase_badge=""
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
    local rolling100_policy_loss="n/a"
    local rolling100_value_loss="n/a"
    local rolling100_entropy="n/a"
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
    local timestamp_display=""
    local remaining_seconds=0
    local cycle_remaining_seconds=0
    local source_label=""
    local source_offset_human=""
    local event_timestamp=""
    local event_message=""
    local recent_event_line=""
    local avg_sync_display="n/a"
    local source_progress_count=0
    local metrics_policy_loss="n/a"
    local metrics_value_loss="n/a"
    local metrics_entropy="n/a"
    local metrics_steps_total="n/a"
    local metrics_learned_episodes="n/a"
    local episode_rate="n/a"
    local step_rate="n/a"
    local intent_display=""
    local source_total_files=0
    local source_completed_files=0
    local source_total_bytes=0
    local source_consumed_bytes=0
    local source_total_human="0 B"
    local source_consumed_human="0 B"
    local source_progress_bar=""
    local source_abs=""
    local source_size=0
    local source_offset_value=0
    local source_percent_display="0"
    local trainer_tail=""
    local trainer_tail_line=""
    local source_read_display="0 B"
    local parent_updates="0"
    local parent_episodes="0"
    local update_delta="0"
    local source_preview=""
    local source_preview_count=0
    local source_preview_more=0
    local monitor_cols=80
    local latest_event_display=""
    local trainer_notice=""
    local metrics_last_line=""
    local live_metric_policy="n/a"
    local live_metric_entropy="n/a"
    local live_metric_steps="n/a"
    local live_metric_episode="0"
    local system_history_abs=""
    local system_cpu_load="n/a"
    local gpu_load="n/a"
    local trainer_cpu_load="n/a"
    local cpu_load_graph=""
    local gpu_load_graph=""
    local trainer_cpu_graph=""
    local load_graph_width=48

    now="$(date +%s)"
    timestamp_display="$(date '+%Y-%m-%d %H:%M:%S')"
    monitor_cols="$(monitor_width)"
    runtime_phase_value="$(runtime_phase)"
    phase_badge="$(monitor_phase_badge "${runtime_phase_value}")"
    phase_display="$(phase_label "${runtime_phase_value}")"
    cycle_display="${CURRENT_CYCLE:-0}"
    model_episodes="$(blacklight_model_episodes "${MODEL_ABS:-}")"
    model_updates="$(blacklight_model_updates "${MODEL_ABS:-}")"
    if [[ -n "${PARENT_MODEL:-}" && -f "${PARENT_MODEL}" ]]; then
        parent_episodes="$(blacklight_model_episodes "${PARENT_MODEL}")"
        parent_updates="$(blacklight_model_updates "${PARENT_MODEL}")"
    fi
    if [[ "${model_updates}" =~ ^[0-9]+$ && "${parent_updates}" =~ ^[0-9]+$ ]] && (( model_updates >= parent_updates )); then
        update_delta="$(( model_updates - parent_updates ))"
    fi
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
        metrics_policy_loss="$(read_summary_value "${METRICS_SUMMARY_ABS}" average_policy_loss)"
        metrics_value_loss="$(read_summary_value "${METRICS_SUMMARY_ABS}" average_value_loss)"
        metrics_entropy="$(read_summary_value "${METRICS_SUMMARY_ABS}" average_entropy)"
        metrics_steps_total="$(read_summary_value "${METRICS_SUMMARY_ABS}" steps_total)"
        metrics_learned_episodes="$(read_summary_value "${METRICS_SUMMARY_ABS}" learned_episodes)"
    fi
    if [[ -f "${METRICS_ABS:-}" ]]; then
        rolling100_win_rate="$(blacklight_read_metric_window "${METRICS_ABS}" 100 2)"
        rolling100_reward="$(blacklight_read_metric_window "${METRICS_ABS}" 100 3)"
        rolling100_distance="$(blacklight_read_metric_window "${METRICS_ABS}" 100 4)"
        rolling100_policy_loss="$(blacklight_read_metric_window "${METRICS_ABS}" 100 14 7 1)"
        rolling100_value_loss="$(blacklight_read_metric_window "${METRICS_ABS}" 100 15 7 1)"
        rolling100_entropy="$(blacklight_read_metric_window "${METRICS_ABS}" 100 16 7 1)"
        rolling500_win_rate="$(blacklight_read_metric_window "${METRICS_ABS}" 500 2)"
        rolling500_reward="$(blacklight_read_metric_window "${METRICS_ABS}" 500 3)"
        rolling500_distance="$(blacklight_read_metric_window "${METRICS_ABS}" 500 4)"
        metrics_last_line="$(tail -n 1 "${METRICS_ABS}")"
        if [[ -n "${metrics_last_line}" && "${metrics_last_line}" == *,* ]]; then
            live_metric_episode="$(printf '%s\n' "${metrics_last_line}" | awk -F',' '{ print $1 }')"
            live_metric_steps="$(printf '%s\n' "${metrics_last_line}" | awk -F',' '{ print $6 }')"
            live_metric_policy="$(printf '%s\n' "${metrics_last_line}" | awk -F',' '{ print $14 }')"
            live_metric_entropy="$(printf '%s\n' "${metrics_last_line}" | awk -F',' '{ print $16 }')"
        fi
    fi
    if [[ "${metrics_episodes:-0}" == "0" && "${live_metric_episode}" != "0" ]]; then
        metrics_episodes="${live_metric_episode}"
    fi
    if [[ "${metrics_policy_loss}" == "n/a" && "${live_metric_policy}" != "n/a" ]]; then
        metrics_policy_loss="${live_metric_policy}"
    fi
    if [[ "${metrics_entropy}" == "n/a" && "${live_metric_entropy}" != "n/a" ]]; then
        metrics_entropy="${live_metric_entropy}"
    fi
    if [[ "${metrics_steps}" == "n/a" && "${live_metric_steps}" != "n/a" ]]; then
        metrics_steps="${live_metric_steps}"
    fi
    if [[ "${metrics_episodes:-0}" == "0" ]]; then
        metrics_win_rate="n/a"
        metrics_reward="n/a"
        metrics_distance="n/a"
        metrics_value="n/a"
        metrics_steps="n/a"
        episode_rate="n/a"
    fi
    if [[ "${TRAINING_MODE:-}" == "teacher" ]]; then
        metrics_win_rate="n/a"
        metrics_reward="n/a"
        metrics_distance="n/a"
        metrics_value="n/a"
        rolling100_win_rate="n/a"
        rolling100_reward="n/a"
        rolling100_distance="n/a"
        rolling500_win_rate="n/a"
        rolling500_reward="n/a"
        rolling500_distance="n/a"
    fi
    if [[ "${metrics_steps_total:-0}" == "0" ]]; then
        step_rate="n/a"
    fi
    if [[ "${metrics_learned_episodes:-0}" == "0" ]]; then
        metrics_policy_loss="n/a"
        metrics_value_loss="n/a"
        metrics_entropy="n/a"
    fi

    if [[ -n "${TRAINING_STARTED_AT:-}" && "${TRAINING_STARTED_AT}" =~ ^[0-9]+$ ]] && (( TRAINING_STARTED_AT > 0 )); then
        elapsed_display="$(blacklight_format_duration $(( now - TRAINING_STARTED_AT )))"
        episode_rate="$(blacklight_format_rate "${metrics_episodes}" "$(( now - TRAINING_STARTED_AT ))" "/s" 3)"
        step_rate="$(blacklight_format_rate "${metrics_steps_total}" "$(( now - TRAINING_STARTED_AT ))" "/s" 3)"
    fi
    if [[ "${metrics_episodes:-0}" == "0" ]]; then
        episode_rate="n/a"
    fi
    if [[ "${metrics_steps_total:-0}" == "0" ]]; then
        step_rate="n/a"
    fi
    if [[ -n "${TRAINING_STARTED_AT:-}" && -n "${TRAINING_END_AT:-}" &&
          "${TRAINING_STARTED_AT}" =~ ^[0-9]+$ && "${TRAINING_END_AT}" =~ ^[0-9]+$ ]] &&
          (( TRAINING_STARTED_AT > 0 && TRAINING_END_AT > TRAINING_STARTED_AT )); then
        total_duration_seconds=$(( TRAINING_END_AT - TRAINING_STARTED_AT ))
        total_duration_display="$(blacklight_format_duration "${total_duration_seconds}")"
        overall_progress_bar="$(monitor_progress_bar $(( now - TRAINING_STARTED_AT )) "${total_duration_seconds}" 30 36)"
        overall_progress_percent="$(blacklight_progress_percent $(( now - TRAINING_STARTED_AT )) "${total_duration_seconds}")"
    fi
    if [[ "${runtime_phase_value}" == "finished" ]]; then
        remaining_display="00:00:00"
    elif [[ "${REPLAY_ONLY:-0}" == "1" ]]; then
        remaining_display="Until replay completes"
    elif [[ -n "${TRAINING_END_AT:-}" && "${TRAINING_END_AT}" =~ ^[0-9]+$ ]] &&
          (( TRAINING_END_AT > TRAINING_STARTED_AT )); then
        remaining_seconds=$(( TRAINING_END_AT - now ))
        if (( remaining_seconds < 0 )); then
            remaining_seconds=0
        fi
        remaining_display="$(blacklight_format_duration "${remaining_seconds}")"
    elif [[ -n "${DURATION_SECONDS:-}" && "${DURATION_SECONDS}" =~ ^[0-9]+$ ]] && (( DURATION_SECONDS > 0 )); then
        remaining_display="Awaiting initial sync"
    fi
    if [[ -n "${CYCLE_DEADLINE:-}" && "${CYCLE_DEADLINE}" =~ ^[0-9]+$ ]] && (( CYCLE_DEADLINE > 0 )); then
        cycle_remaining_seconds=$(( CYCLE_DEADLINE - now ))
        if (( cycle_remaining_seconds < 0 )); then
            cycle_remaining_seconds=0
        fi
        cycle_remaining_display="$(blacklight_format_duration "${cycle_remaining_seconds}")"
    fi
    if [[ -n "${SYNC_STARTED_AT:-}" && "${SYNC_STARTED_AT}" =~ ^[0-9]+$ && "${runtime_phase_value}" == "syncing" ]] && (( SYNC_STARTED_AT > 0 )); then
        sync_elapsed_display="$(blacklight_format_duration $(( now - SYNC_STARTED_AT )))"
        sync_activity_bar="$(monitor_activity_bar $(( now - SYNC_STARTED_AT )) 26 36)"
    fi
    if [[ -n "${LAST_SYNC_DURATION:-}" && "${LAST_SYNC_DURATION}" =~ ^[0-9]+$ ]]; then
        last_sync_display="$(blacklight_format_duration "${LAST_SYNC_DURATION}")"
    fi
    if [[ "${avg_sync_seconds}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        avg_sync_display="${avg_sync_seconds}s"
    fi
    if [[ -f "${EVENTS_LOG_ABS:-}" ]]; then
        recent_events="$(tail -n 6 "${EVENTS_LOG_ABS}")"
        latest_event_display="$(printf '%s\n' "${recent_events}" | tail -n 1)"
        latest_event_display="${latest_event_display#*] }"
    fi
    if [[ -n "${PARALLEL_WORKERS:-}" && "${PARALLEL_WORKERS}" =~ ^[0-9]+$ ]] && (( PARALLEL_WORKERS > 0 )); then
        worker_bar="$(monitor_progress_bar "${running_workers}" "${PARALLEL_WORKERS}" 24 32)"
        worker_percent="$(blacklight_progress_percent "${running_workers}" "${PARALLEL_WORKERS}")"
    fi
    if [[ "${runtime_phase_value}" == "collecting" &&
          -n "${CYCLE_DEADLINE:-}" && "${CYCLE_DEADLINE}" =~ ^[0-9]+$ ]]; then
        cycle_start_at="${LAST_SYNC_FINISHED_AT:-${TRAINING_STARTED_AT:-0}}"
        if [[ "${cycle_start_at}" =~ ^[0-9]+$ ]] && (( CYCLE_DEADLINE > cycle_start_at )); then
            cycle_total_seconds=$(( CYCLE_DEADLINE - cycle_start_at ))
            cycle_elapsed_seconds=$(( now - cycle_start_at ))
            cycle_progress_bar="$(monitor_progress_bar "${cycle_elapsed_seconds}" "${cycle_total_seconds}" 24 33)"
            cycle_progress_percent="$(blacklight_progress_percent "${cycle_elapsed_seconds}" "${cycle_total_seconds}")"
        fi
    fi

    if [[ "${COLLECT_ONLY:-0}" == "1" ]]; then
        intent_display="Capturing teacher examples only. This run does not update the model weights."
    elif [[ "${REPLAY_ONLY:-0}" == "1" ]]; then
        intent_display="Replaying teacher logs into the CNN. No worker matches are running in this phase."
    else
        intent_display="Alternating between collection windows and sync passes."
    fi

    if [[ -n "${SOURCE_LIST_ABS:-}" && -f "${SOURCE_LIST_ABS}" ]]; then
        while IFS= read -r source_path; do
            [[ -n "${source_path}" ]] || continue
            source_total_files=$(( source_total_files + 1 ))
            source_abs="${REPO_ROOT}/var/${source_path}"
            source_size="$(blacklight_file_size "${source_abs}")"
            source_total_bytes=$(( source_total_bytes + source_size ))
        done < "${SOURCE_LIST_ABS}"

        if [[ -n "${STATE_FILE_ABS:-}" && -f "${STATE_FILE_ABS}" ]]; then
            while IFS='|' read -r source_path source_episodes source_offset; do
                [[ -n "${source_path}" ]] || continue
                source_abs="${REPO_ROOT}/var/${source_path}"
                source_size="$(blacklight_file_size "${source_abs}")"
                source_offset_value="${source_offset:-0}"
                if [[ -z "${source_offset_value}" || ! "${source_offset_value}" =~ ^[0-9]+$ ]]; then
                    source_offset_value=0
                fi
                if (( source_size > 0 && source_offset_value > source_size )); then
                    source_offset_value="${source_size}"
                fi
                source_consumed_bytes=$(( source_consumed_bytes + source_offset_value ))
                if (( source_size > 0 && source_offset_value >= source_size )); then
                    source_completed_files=$(( source_completed_files + 1 ))
                fi
            done < <(blacklight_format_source_progress "${STATE_FILE_ABS}" "${SOURCE_LIST_ABS}")
        fi

        source_total_human="$(blacklight_human_bytes "${source_total_bytes}")"
        source_consumed_human="$(blacklight_human_bytes "${source_consumed_bytes}")"
        if (( source_total_bytes > 0 )); then
            source_progress_bar="$(monitor_progress_bar "${source_consumed_bytes}" "${source_total_bytes}" 24 35)"
            source_progress_percent="$(blacklight_progress_percent "${source_consumed_bytes}" "${source_total_bytes}")"
        fi

        while IFS= read -r source_path; do
            [[ -n "${source_path}" ]] || continue
            source_label="$(monitor_source_label "${source_path}")"
            if (( source_preview_count < 3 )); then
                if [[ -n "${source_preview}" ]]; then
                    source_preview="${source_preview}, "
                fi
                source_preview="${source_preview}${source_label}"
            else
                source_preview_more=$(( source_preview_more + 1 ))
            fi
            source_preview_count=$(( source_preview_count + 1 ))
        done < "${SOURCE_LIST_ABS}"
        if (( source_preview_more > 0 )); then
            source_preview="${source_preview} +${source_preview_more} more"
        fi
    fi

    if [[ "${REPLAY_ONLY:-0}" == "1" && ( -z "${STATE_FILE_ABS:-}" || ! -f "${STATE_FILE_ABS}" ) ]]; then
        source_read_display="Offsets pending"
    else
        source_read_display="${source_consumed_human}"
    fi

    if [[ -f "${TRAINER_LOG_ABS:-}" ]]; then
        trainer_tail="$(tail -n 6 "${TRAINER_LOG_ABS}")"
        while IFS= read -r trainer_tail_line; do
            trainer_tail_line="${trainer_tail_line#\[0\] }"
            trainer_tail_line="${trainer_tail_line#"${trainer_tail_line%%[![:space:]]*}"}"
            [[ -n "${trainer_tail_line}" ]] || continue
            case "${trainer_tail_line}" in
                *unknown*|*Unknown*|*failed*|*Failed*|*error*|*Error*|*disabled*|*Disabled*)
                    if [[ -n "${trainer_notice}" ]]; then
                        trainer_notice="${trainer_notice} | "
                    fi
                    trainer_notice="${trainer_notice}${trainer_tail_line}"
                    ;;
            esac
        done <<< "${trainer_tail}"
    fi

    if [[ -n "${RUN_DIR_ABS:-}" ]]; then
        system_history_abs="${RUN_DIR_ABS}/monitor_system_load.tsv"
    else
        system_history_abs="${REPO_ROOT}/var/monitor_system_load.tsv"
    fi
    system_cpu_load="$(monitor_system_cpu_percent || true)"
    gpu_load="$(monitor_gpu_percent || true)"
    trainer_cpu_load="$(monitor_trainer_cpu_percent "${TRAINER_PID:-0}" || true)"
    [[ "${system_cpu_load}" =~ ^[0-9]+([.][0-9]+)?$ ]] || system_cpu_load="n/a"
    [[ "${gpu_load}" =~ ^[0-9]+([.][0-9]+)?$ ]] || gpu_load="n/a"
    [[ "${trainer_cpu_load}" =~ ^[0-9]+([.][0-9]+)?$ ]] || trainer_cpu_load="n/a"
    monitor_append_load_sample "${system_history_abs}" "${now}" "${system_cpu_load}" "${gpu_load}" "${trainer_cpu_load}"

    load_graph_width=$(( monitor_cols - 34 ))
    if (( load_graph_width < 24 )); then
        load_graph_width=24
    elif (( load_graph_width > 72 )); then
        load_graph_width=72
    fi
    cpu_load_graph="$(monitor_load_sparkline "${system_history_abs}" cpu "${load_graph_width}")"
    gpu_load_graph="$(monitor_load_sparkline "${system_history_abs}" gpu "${load_graph_width}")"
    trainer_cpu_graph="$(monitor_load_sparkline "${system_history_abs}" trainer "${load_graph_width}")"

    monitor_heading "Blacklight Ops  ${timestamp_display}"

    monitor_section "State"
    monitor_detail_row "Run" "${RUN_NAME:-unknown}"
    monitor_detail_row "Phase" "${phase_badge} ${phase_display}"
    monitor_detail_row "Intent" "$(monitor_fit_value "${intent_display}" 18)"
    monitor_compact_row "Mode" "${TRAINING_MODE:-unknown}" "Profile" "${PROFILE:-none}" "Generation" "${GENERATION:-unknown}"
    monitor_compact_row "Elapsed" "${elapsed_display}" "Remaining" "${remaining_display}" "Update +" "${update_delta}"
    if [[ "${REPLAY_ONLY:-0}" == "1" ]]; then
        if [[ -f "${STATE_FILE_ABS:-}" && -n "${source_progress_bar}" ]]; then
            monitor_detail_row "Replay" "${source_progress_bar} ${source_progress_percent}% • ${source_read_display} of ${source_total_human}"
        else
            monitor_detail_row "Replay" "Trainer live • offset snapshots pending • ${source_total_human} queued"
        fi
        if [[ -n "${sync_activity_bar}" ]]; then
            monitor_detail_row "Trainer" "${sync_activity_bar} ${sync_elapsed_display} • PID ${TRAINER_PID:-0}"
        else
            monitor_detail_row "Trainer" "Idle • PID ${TRAINER_PID:-0}"
        fi
    elif [[ "${runtime_phase_value}" == "collecting" ]]; then
        if [[ -n "${overall_progress_bar}" ]]; then
            monitor_detail_row "Run" "${overall_progress_bar} ${overall_progress_percent}% of ${total_duration_display}"
        fi
        if [[ -n "${cycle_progress_bar}" ]]; then
            monitor_detail_row "Window" "${cycle_progress_bar} ${cycle_progress_percent}% • eta ${cycle_remaining_display}"
        fi
        if [[ -n "${worker_bar}" ]]; then
            monitor_detail_row "Workers" "${worker_bar} ${worker_percent}% • ${running_workers}/${PARALLEL_WORKERS:-0} active"
        fi
    else
        if [[ -n "${overall_progress_bar}" ]]; then
            monitor_detail_row "Run" "${overall_progress_bar} ${overall_progress_percent}% of ${total_duration_display}"
        else
            monitor_detail_row "Run" "Open-ended"
        fi
    fi

    monitor_section "Signals"
    monitor_compact_row "Updates" "${model_updates} (+${update_delta})" "Model Ep" "${model_episodes}" "CSV Ep" "${metrics_episodes:-0}"
    monitor_compact_row "Policy" "${metrics_policy_loss:-n/a}" "Entropy" "${metrics_entropy:-n/a}" "Step/ep" "${metrics_steps:-n/a}"
    monitor_compact_row "Learned" "${metrics_learned_episodes:-n/a}" "Last Sync" "${last_sync_display}" "Avg Sync" "${avg_sync_display}"
    if [[ "${COLLECT_ONLY:-0}" == "1" ]]; then
        monitor_compact_row "Teacher" "${source_total_human}" "New Data" "$(blacklight_human_bytes "${experience_growth_bytes}")" "Workers" "${running_workers}/${PARALLEL_WORKERS:-0}"
    elif [[ "${REPLAY_ONLY:-0}" == "1" ]]; then
        monitor_compact_row "Teacher" "${source_total_human}" "Replay" "${source_read_display}" "Trainer PID" "${TRAINER_PID:-0}"
    fi
    if [[ "${TRAINING_MODE:-}" != "teacher" ]]; then
        monitor_compact_row "Win Rate" "${metrics_win_rate:-n/a}" "Avg Reward" "${metrics_reward:-n/a}" "Avg Dist" "${metrics_distance:-n/a}"
    fi

    monitor_section "System Load"
    monitor_load_row "CPU total" "${system_cpu_load}" "${cpu_load_graph}" "system"
    monitor_load_row "Trainer CPU" "${trainer_cpu_load}" "${trainer_cpu_graph}" "trainer"
    monitor_load_row "GPU load" "${gpu_load}" "${gpu_load_graph}" "device"
    if (( source_total_files > 0 )); then
        monitor_detail_row "Teacher Data" "${source_total_human} • ${source_total_files} logs"
    fi
    if [[ -n "${latest_event_display}" ]]; then
        monitor_detail_row "Last Event" "$(monitor_fit_value "${latest_event_display}" 18)"
    fi
    if [[ -n "${trainer_notice}" ]]; then
        monitor_detail_row "Warnings" "$(monitor_fit_value "${trainer_notice}" 18)"
    fi

    if [[ -n "${recent_events}" ]]; then
        monitor_section "Events"
        if (( monitor_cols < 100 )); then
            printf '%s %s\n' "$(monitor_colorize '2;37' "$(monitor_rail)")" "$(monitor_fit_value "${latest_event_display}" 4)"
            return 0
        fi
        while IFS= read -r recent_event_line; do
            [[ -n "${recent_event_line}" ]] || continue
            event_timestamp="$(printf '%s\n' "${recent_event_line}" | sed -n 's/^\[\([^]]*\)\].*/\1/p')"
            event_message="${recent_event_line#*] }"
            if [[ -n "${event_timestamp}" ]]; then
                event_timestamp="${event_timestamp#* }"
                printf '%s %-10s %s\n' "$(monitor_colorize '2;37' "$(monitor_rail)")" "${event_timestamp}" "${event_message}"
            else
                printf '%s %s\n' "$(monitor_colorize '2;37' "$(monitor_rail)")" "${recent_event_line}"
            fi
        done <<< "${recent_events}"
    else
        monitor_section "Events"
        printf '%s no events yet\n' "$(monitor_colorize '2;37' "$(monitor_rail)")"
    fi
}

run_monitor() {
    local interval="2"
    local once="0"
    local frame=""
    local interactive="0"
    local classic="0"
    local no_color="0"
    local dashboard_script="${SCRIPT_DIR}/blacklight_terminal_dashboard.py"
    local python_bin=""
    local -a dashboard_args=()

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
            --classic)
                classic="1"
                shift
                ;;
            --no-color)
                no_color="1"
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

    if [[ "${classic}" != "1" && "${BLACKLIGHT_MONITOR_CLASSIC:-0}" != "1" && -f "${dashboard_script}" ]] &&
       python_bin="$(command -v python3 2>/dev/null)"; then
        dashboard_args=("${dashboard_script}" "--repo-root" "${REPO_ROOT}" "--interval" "${interval}")
        if [[ "${once}" == "1" ]]; then
            dashboard_args+=("--once")
        fi
        if [[ "${no_color}" == "1" ]]; then
            dashboard_args+=("--no-color")
        fi
        "${python_bin}" "${dashboard_args[@]}"
        return $?
    fi

    if [[ "${once}" == "0" ]] && monitor_supports_repaint; then
        interactive="1"
        printf '\033[?1049h\033[H\033[2J\033[?25l'
        trap 'printf '\''\033[?25h\033[?1049l'\''' EXIT INT TERM
    fi

    while :; do
        if [[ "${interactive}" == "1" || ( -t 1 && -z "${NO_COLOR:-}" ) ]]; then
            frame="$(BLACKLIGHT_MONITOR_TTY_COLOR=1 render_monitor)"
        else
            frame="$(render_monitor)"
        fi
        if [[ "${interactive}" == "1" ]]; then
            printf '\033[?25l\033[H\033[2J%s' "${frame}"
        else
            printf '%s\n' "${frame}"
        fi
        if [[ "${once}" == "1" ]]; then
            break
        fi
        sleep "${interval}"
    done

    if [[ "${interactive}" == "1" ]]; then
        printf '\033[?25h\033[?1049l'
        trap - EXIT INT TERM
    fi
}

run_dashboard() {
    local host="127.0.0.1"
    local port="8765"
    local run_name=""
    local open_browser="1"
    local -a dashboard_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --host)
                host="$2"
                shift 2
                ;;
            --port)
                port="$2"
                shift 2
                ;;
            --run)
                run_name="$2"
                shift 2
                ;;
            --no-open)
                open_browser="0"
                shift
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                echo "Unknown dashboard option: $1" >&2
                usage >&2
                exit 1
                ;;
        esac
    done

    if ! [[ "${port}" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
        echo "Dashboard port must be between 1 and 65535." >&2
        exit 1
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "python3 is required for the Blacklight dashboard." >&2
        exit 1
    fi

    dashboard_args=( "${SCRIPT_DIR}/blacklight_dashboard_ops.py" --repo-root "${REPO_ROOT}" --host "${host}" --port "${port}" )
    if [[ -n "${run_name}" ]]; then
        dashboard_args+=( --run "${run_name}" )
    fi
    if [[ "${open_browser}" == "1" ]]; then
        dashboard_args+=( --open )
        exec python3 "${dashboard_args[@]}"
    fi

    exec python3 "${dashboard_args[@]}"
}

run_pipeline() {
    local pipeline_doc="${REPO_ROOT}/docs/BLACKLIGHT_TRAINING_PIPELINE.md"

    if [[ ! -f "${pipeline_doc}" ]]; then
        echo "Missing pipeline document: ${pipeline_doc}" >&2
        exit 1
    fi

    cat "${pipeline_doc}"
}

run_collect() {
    local profile=""
    local duration=""
    local rounds=""
    local resume_model=""
    local generation="blacklight_collect"
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
    local fresh_mode="0"

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
            --resume)
                resume_model="$2"
                shift 2
                ;;
            --fresh)
                fresh_mode="1"
                shift
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
        training_mode="parallel"
    fi

    if [[ "${fresh_mode}" != "1" && -n "${resume_model}" ]]; then
        if ! resolved_resume_model="$(blacklight_resolve_model_input "${resume_model}" 1)"; then
            echo "Could not resolve resume model: ${resume_model}" >&2
            exit 1
        fi
    elif [[ "${fresh_mode}" != "1" ]]; then
        resolved_resume_model="$(blacklight_best_model || true)"
    fi
    if [[ -n "${resolved_resume_model}" ]]; then
        export ARMAGETRON_SELFPLAY_INITIAL_MODEL="${resolved_resume_model}"
        export ARMAGETRON_SELFPLAY_PARENT_MODEL="${resolved_resume_model}"
    fi

    if [[ -n "${duration}" ]]; then
        export ARMAGETRON_SELFPLAY_DURATION_SECONDS="${duration}"
    fi
    if [[ -n "${rounds}" ]]; then
        export ARMAGETRON_SELFPLAY_LIMIT_ROUNDS="${rounds}"
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
    export ARMAGETRON_SELFPLAY_REPLAY_ONLY="0"
    export ARMAGETRON_SELFPLAY_COLLECT_ONLY="1"
    export ARMAGETRON_SELFPLAY_REPLAY_SOURCE_LIST_ABS=""

    if [[ "${training_mode}" == "parallel" ]]; then
        if [[ "${fast_mode}" == "1" ]]; then
            echo "Teacher data collection needs full worker teacher logs, so --fast cannot be combined with collect." >&2
            exit 1
        fi
        exec "${SCRIPT_DIR}/train_neural_ai_parallel.sh"
    fi

    echo "Unsupported collection mode: ${training_mode}" >&2
    exit 1
}

finish_in_place_training() {
    local source_label="$1"
    local trained_model=""
    local trained_run_dir=""
    local installed_model=""

    if ! blacklight_source_latest_manifest; then
        echo "In-place training finished without a latest run manifest." >&2
        exit 1
    fi

    trained_model="${MODEL_ABS:-}"
    trained_run_dir="${RUN_DIR_ABS:-}"
    if ! blacklight_model_looks_valid "${trained_model}"; then
        echo "In-place training did not produce a valid model: ${trained_model}" >&2
        exit 1
    fi

    installed_model="$(blacklight_install_current_model "${trained_model}" "${source_label}" "in-place")"

    if [[ -n "${trained_run_dir}" && "${trained_run_dir}" == "${REPO_ROOT}/var/blacklight_runs/"* ]]; then
        rm -rf "${trained_run_dir}"
    fi
    rm -f "${BLACKLIGHT_LATEST_MANIFEST_PATH}"

    echo "Blacklight in-place training applied"
    echo "current_model ${installed_model}"
    echo "temporary_run_deleted ${trained_run_dir}"
}

run_train() {
    local data_source=""
    local profile=""
    local resume_model=""
    local checkpoint_every=""
    local duration=""
    local rounds=""
    local generation="blacklight_train"
    local run_name=""
    local bin_path=""
    local fast_mode="0"
    local heavy_mode="0"
    local parallel_workers=""
    local sync_seconds=""
    local resolved_resume_model=""
    local resolved_source_list=""
    local fresh_mode="0"
    local training_mode=""
    local profile_value=""
    local gpu_mode="${BLACKLIGHT_TRAIN_GPU_MODE:-1}"
    local in_place="0"
    local gpu_device="auto"
    local gpu_batch_size=""
    local gpu_epochs=""
    local gpu_lr=""
    local gpu_max_examples=""
    local gpu_python=""
    local -a gpu_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --data)
                data_source="$2"
                shift 2
                ;;
            --profile)
                profile="$2"
                shift 2
                ;;
            --resume)
                resume_model="$2"
                shift 2
                ;;
            --fresh)
                fresh_mode="1"
                shift
                ;;
            --gpu)
                gpu_mode="1"
                shift
                ;;
            --cpu)
                gpu_mode="0"
                shift
                ;;
            --in-place|--current)
                in_place="1"
                shift
                ;;
            --device)
                gpu_device="$2"
                shift 2
                ;;
            --batch-size)
                gpu_batch_size="$2"
                shift 2
                ;;
            --epochs)
                gpu_epochs="$2"
                shift 2
                ;;
            --lr)
                gpu_lr="$2"
                shift 2
                ;;
            --max-examples)
                gpu_max_examples="$2"
                shift 2
                ;;
            --checkpoint-every)
                checkpoint_every="$2"
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

    if [[ -n "${profile}" && "${profile}" != "teacher" ]]; then
        if [[ "${in_place}" == "1" ]]; then
            echo "--in-place trains from collected teacher data only. Run collect first, then train --in-place." >&2
            exit 1
        fi
        if [[ -n "${data_source}" ]]; then
            echo "--data is only used by the teacher replay train path; profile training collects worker data itself." >&2
            exit 1
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
            [[ -n "${parallel_workers}" && -n "${profile_value}" ]] && export ARMAGETRON_SELFPLAY_PARALLEL_RECORD_STRIDE="${profile_value}"
            profile_value="$(blacklight_train_profile_value "${profile}" save_every)"
            [[ -n "${profile_value}" && -z "${ARMAGETRON_SELFPLAY_SAVE_EVERY:-}" ]] && export ARMAGETRON_SELFPLAY_SAVE_EVERY="${profile_value}"
            profile_value="$(blacklight_train_profile_value "${profile}" checkpoint_every)"
            [[ -n "${profile_value}" && -z "${checkpoint_every}" && -z "${ARMAGETRON_SELFPLAY_CHECKPOINT_EVERY:-}" ]] && export ARMAGETRON_SELFPLAY_CHECKPOINT_EVERY="${profile_value}"
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
            training_mode="parallel"
        fi

        if [[ "${fresh_mode}" != "1" && -n "${resume_model}" ]]; then
            if ! resolved_resume_model="$(blacklight_resolve_model_input "${resume_model}" 1)"; then
                echo "Could not resolve resume model: ${resume_model}" >&2
                exit 1
            fi
        elif [[ "${fresh_mode}" != "1" ]]; then
            resolved_resume_model="$(blacklight_best_model || true)"
        fi

        if [[ -n "${resolved_resume_model}" ]]; then
            export ARMAGETRON_SELFPLAY_INITIAL_MODEL="${resolved_resume_model}"
            export ARMAGETRON_SELFPLAY_PARENT_MODEL="${resolved_resume_model}"
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
        export ARMAGETRON_SELFPLAY_REPLAY_ONLY="0"
        export ARMAGETRON_SELFPLAY_COLLECT_ONLY="0"
        export ARMAGETRON_SELFPLAY_REPLAY_SOURCE_LIST_ABS=""

        if [[ "${training_mode}" == "parallel" ]]; then
            exec "${SCRIPT_DIR}/train_neural_ai_parallel.sh"
        fi

        echo "Unsupported training mode: ${training_mode}" >&2
        exit 1
    fi

    if ! resolved_source_list="$(blacklight_resolve_source_list_input "${data_source}")"; then
        echo "Could not resolve teacher data source." >&2
        echo "Run ./scripts/blacklight.sh collect first, or pass --data SOURCE_LIST_OR_RUN." >&2
        exit 1
    fi

    if [[ "${fresh_mode}" != "1" && -n "${resume_model}" ]]; then
        if ! resolved_resume_model="$(blacklight_resolve_model_input "${resume_model}" 1)"; then
            echo "Could not resolve resume model: ${resume_model}" >&2
            exit 1
        fi
    elif [[ "${fresh_mode}" != "1" ]]; then
        resolved_resume_model="$(blacklight_best_model || true)"
    fi

    if [[ "${gpu_mode}" != "0" ]]; then
        if [[ -z "${resolved_resume_model}" ]]; then
            echo "GPU teacher replay needs an initial model. Use --resume champion or omit --fresh." >&2
            exit 1
        fi
        if [[ -n "${BLACKLIGHT_GPU_PYTHON:-}" ]]; then
            gpu_python="${BLACKLIGHT_GPU_PYTHON}"
        elif [[ -x "${REPO_ROOT}/.venv-blacklight-gpu/bin/python" ]]; then
            gpu_python="${REPO_ROOT}/.venv-blacklight-gpu/bin/python"
        else
            gpu_python="$(command -v python3 || true)"
        fi
        if [[ -z "${gpu_python}" ]]; then
            echo "Could not find Python for GPU training." >&2
            exit 1
        fi

        gpu_args=(
            "${SCRIPT_DIR}/blacklight_gpu_teacher_train.py"
            "--repo-root" "${REPO_ROOT}"
            "--source-list" "${resolved_source_list}"
            "--initial-model" "${resolved_resume_model}"
            "--parent-model" "${resolved_resume_model}"
            "--generation" "${generation}"
            "--device" "${gpu_device}"
        )
        if [[ -n "${run_name}" ]]; then
            gpu_args+=("--run-name" "${run_name}")
        fi
        if [[ -n "${checkpoint_every}" ]]; then
            gpu_args+=("--checkpoint-every" "${checkpoint_every}")
        fi
        if [[ -n "${gpu_batch_size}" ]]; then
            gpu_args+=("--batch-size" "${gpu_batch_size}")
        fi
        if [[ -n "${gpu_epochs}" ]]; then
            gpu_args+=("--epochs" "${gpu_epochs}")
        fi
        if [[ -n "${gpu_lr}" ]]; then
            gpu_args+=("--lr" "${gpu_lr}")
        fi
        if [[ -n "${gpu_max_examples}" ]]; then
            gpu_args+=("--max-examples" "${gpu_max_examples}")
        fi
        if [[ "${in_place}" == "1" ]]; then
            "${gpu_python}" "${gpu_args[@]}"
            finish_in_place_training "in-place-gpu"
            return 0
        fi

        exec "${gpu_python}" "${gpu_args[@]}"
    fi

    export ARMAGETRON_SELFPLAY_PROFILE="teacher"
    export ARMAGETRON_SELFPLAY_REPLAY_ONLY="1"
    export ARMAGETRON_SELFPLAY_COLLECT_ONLY="0"
    export ARMAGETRON_SELFPLAY_REPLAY_SOURCE_LIST_ABS="${resolved_source_list}"
    export ARMAGETRON_SELFPLAY_GENERATION="${generation}"

    if [[ -n "${resolved_resume_model}" ]]; then
        export ARMAGETRON_SELFPLAY_INITIAL_MODEL="${resolved_resume_model}"
        export ARMAGETRON_SELFPLAY_PARENT_MODEL="${resolved_resume_model}"
    else
        export ARMAGETRON_SELFPLAY_INITIAL_MODEL=""
        export ARMAGETRON_SELFPLAY_PARENT_MODEL=""
    fi

    if [[ -n "${checkpoint_every}" ]]; then
        export ARMAGETRON_SELFPLAY_CHECKPOINT_EVERY="${checkpoint_every}"
    fi
    if [[ -n "${run_name}" ]]; then
        export ARMAGETRON_SELFPLAY_RUN_NAME="${run_name}"
    fi
    if [[ -n "${bin_path}" ]]; then
        export ARMAGETRON_SELFPLAY_BIN="${bin_path}"
    fi

    if [[ "${in_place}" == "1" ]]; then
        "${SCRIPT_DIR}/train_neural_ai_parallel.sh"
        finish_in_place_training "in-place-cpu"
        return 0
    fi

    exec "${SCRIPT_DIR}/train_neural_ai_parallel.sh"
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
    local sessions=""
    local duration=""
    local rounds=""
    local bench_args=()

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
            --sessions)
                sessions="$2"
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

    bench_args=( --suite "${suite}" --candidate "${candidate}" )
    if [[ -n "${reference}" ]]; then
        bench_args+=( --reference "${reference}" )
    fi
    if [[ -n "${sessions}" ]]; then
        bench_args+=( --sessions "${sessions}" )
    fi
    if [[ -n "${duration}" ]]; then
        bench_args+=( --duration "${duration}" )
    fi
    if [[ -n "${rounds}" ]]; then
        bench_args+=( --rounds "${rounds}" )
    fi

    exec "${SCRIPT_DIR}/benchmark_blacklight.sh" "${bench_args[@]}"
}

run_promote() {
    local candidate="latest"
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

    if [[ -n "${bin_path}" ]]; then
        export ARMAGETRON_SELFPLAY_BIN="${bin_path}"
    fi

    exec "${SCRIPT_DIR}/promote_blacklight.sh" --candidate "${candidate}"
}

run_champion() {
    local action="show"
    local model_source="latest"
    local model_abs=""
    local replace="0"
    local champion_model=""

    if [[ $# -gt 0 ]]; then
        case "$1" in
            show|init)
                action="$1"
                shift
                ;;
        esac
    fi

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)
                model_source="$2"
                shift 2
                ;;
            --replace)
                replace="1"
                shift
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                echo "Unknown champion option: $1" >&2
                usage >&2
                exit 1
                ;;
        esac
    done

    case "${action}" in
        show)
            echo "Blacklight champion"
            if champion_model="$(blacklight_current_champion_model)"; then
                echo "champion_model ${champion_model}"
                echo "champion_registry ${BLACKLIGHT_CHAMPION_ENV_PATH}"
                if [[ -f "${BLACKLIGHT_CHAMPION_MODEL_PATH}" ]]; then
                    echo "champion_active_model ${BLACKLIGHT_CHAMPION_MODEL_PATH}"
                fi
            else
                echo "champion_model none"
                echo "next_step ./scripts/blacklight.sh champion init --model latest"
            fi
            if model_abs="$(blacklight_latest_candidate_model 2>/dev/null)"; then
                echo "latest_training_model ${model_abs}"
            fi
            ;;
        init)
            if [[ "${replace}" != "1" ]] && blacklight_current_champion_model >/dev/null 2>&1; then
                echo "A champion already exists. Use --replace only if you intentionally want to reset the champion." >&2
                exit 1
            fi
            if ! model_abs="$(blacklight_resolve_model_input "${model_source}" 1)"; then
                echo "Could not resolve champion model: ${model_source}" >&2
                exit 1
            fi
            champion_model="$(blacklight_install_champion_model "${model_abs}" "${model_source}" "" "" "initial")"
            echo "Blacklight champion initialized"
            echo "source_model ${model_abs}"
            echo "champion_model ${champion_model}"
            echo "champion_active_model ${BLACKLIGHT_CHAMPION_MODEL_PATH}"
            echo "champion_registry ${BLACKLIGHT_CHAMPION_ENV_PATH}"
            ;;
        *)
            echo "Unknown champion action: ${action}" >&2
            usage >&2
            exit 1
            ;;
    esac
}

run_current() {
    local action="show"
    local model_source="latest"
    local model_abs=""
    local current_model=""
    local installed_model=""
    local alpha="0.10"
    local run_name=""
    local output_only="0"
    local distill_run_dir=""
    local distill_output_model=""
    local distill_summary=""
    local python_bin=""
    local source_arg=""
    local source_abs=""
    local -a source_args=()
    local -a source_models=()
    local -a distill_args=()

    if [[ $# -gt 0 ]]; then
        case "$1" in
            show|set|distill)
                action="$1"
                shift
                ;;
        esac
    fi

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)
                model_source="$2"
                shift 2
                ;;
            --from|--source)
                source_args+=( "$2" )
                shift 2
                ;;
            --alpha)
                alpha="$2"
                shift 2
                ;;
            --name)
                run_name="$2"
                shift 2
                ;;
            --output-only)
                output_only="1"
                shift
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                echo "Unknown current option: $1" >&2
                usage >&2
                exit 1
                ;;
        esac
    done

    case "${action}" in
        show)
            echo "Blacklight current best"
            if current_model="$(blacklight_current_champion_model)"; then
                echo "current_model ${current_model}"
                echo "current_registry ${BLACKLIGHT_CHAMPION_ENV_PATH}"
                if [[ -f "${BLACKLIGHT_CHAMPION_MODEL_PATH}" ]]; then
                    echo "active_model ${BLACKLIGHT_CHAMPION_MODEL_PATH}"
                fi
            else
                echo "current_model none"
                echo "next_step ./scripts/blacklight.sh current set --model latest"
            fi
            if model_abs="$(blacklight_latest_candidate_model 2>/dev/null)"; then
                echo "latest_training_model ${model_abs}"
            fi
            ;;
        set)
            if ! model_abs="$(blacklight_resolve_model_input "${model_source}" 1)"; then
                echo "Could not resolve current-best model: ${model_source}" >&2
                exit 1
            fi
            installed_model="$(blacklight_install_current_model "${model_abs}" "${model_source}" "manual")"
            echo "Blacklight current best updated"
            echo "source_model ${model_abs}"
            echo "current_model ${installed_model}"
            echo "active_model ${BLACKLIGHT_CHAMPION_MODEL_PATH}"
            echo "current_registry ${BLACKLIGHT_CHAMPION_ENV_PATH}"
            if blacklight_source_champion_registry >/dev/null 2>&1; then
                echo "backup_model ${CURRENT_BACKUP_MODEL_ABS:-}"
            fi
            ;;
        distill)
            if ! current_model="$(blacklight_current_champion_model)"; then
                echo "No current Blacklight model exists. Set one before distilling into it." >&2
                exit 1
            fi
            if (( ${#source_args[@]} == 0 )); then
                echo "current distill needs at least one --from MODEL source." >&2
                exit 1
            fi
            for source_arg in "${source_args[@]}"; do
                if ! source_abs="$(blacklight_resolve_model_input "${source_arg}" 1)"; then
                    echo "Could not resolve distillation source: ${source_arg}" >&2
                    exit 1
                fi
                if [[ "${source_abs}" == "${current_model}" ]]; then
                    echo "Skipping source because it is already the current model: ${source_arg}" >&2
                    continue
                fi
                source_models+=( "${source_abs}" )
            done
            if (( ${#source_models[@]} == 0 )); then
                echo "No distinct distillation sources remain after resolving inputs." >&2
                exit 1
            fi
            if [[ -z "${run_name}" ]]; then
                run_name="blacklight_distill_$(date +%Y%m%d-%H%M%S)"
            fi
            distill_run_dir="${REPO_ROOT}/var/blacklight_runs/${run_name}"
            distill_output_model="${distill_run_dir}/distilled_current_model.txt"
            distill_summary="${distill_run_dir}/distill_summary.env"
            python_bin="$(command -v python3 || true)"
            if [[ -z "${python_bin}" ]]; then
                echo "Could not find python3 for current-model distillation." >&2
                exit 1
            fi
            mkdir -p "${distill_run_dir}"
            distill_args=(
                "${SCRIPT_DIR}/blacklight_distill_models.py"
                "--current" "${current_model}"
                "--output" "${distill_output_model}"
                "--summary" "${distill_summary}"
                "--alpha" "${alpha}"
            )
            for source_abs in "${source_models[@]}"; do
                distill_args+=( "--source" "${source_abs}" )
            done

            "${python_bin}" "${distill_args[@]}"

            if [[ "${output_only}" == "1" ]]; then
                echo "Blacklight distilled current model written"
                echo "current_model ${current_model}"
                echo "distilled_model ${distill_output_model}"
                echo "distill_summary ${distill_summary}"
                echo "not_installed 1"
                return 0
            fi

            installed_model="$(blacklight_install_current_model "${distill_output_model}" "current-distill" "distilled-current")"
            echo "Blacklight current champion distilled"
            echo "previous_current ${current_model}"
            echo "current_model ${installed_model}"
            echo "distilled_model ${distill_output_model}"
            echo "distill_summary ${distill_summary}"
            echo "alpha ${alpha}"
            echo "sources ${#source_models[@]}"
            if blacklight_source_champion_registry >/dev/null 2>&1; then
                echo "backup_model ${CURRENT_BACKUP_MODEL_ABS:-}"
            fi
            ;;
        *)
            echo "Unknown current action: ${action}" >&2
            usage >&2
            exit 1
            ;;
    esac
}

run_gpu_setup() {
    local venv_path="${REPO_ROOT}/.venv-blacklight-gpu"
    local python_bin="${venv_path}/bin/python"

    if ! command -v uv >/dev/null 2>&1; then
        echo "GPU setup needs uv. Install uv or set BLACKLIGHT_GPU_PYTHON to a Python that already has PyTorch." >&2
        exit 1
    fi

    if [[ ! -x "${python_bin}" ]]; then
        echo "Creating Blacklight GPU Python environment at ${venv_path}"
        uv venv --python 3.12 "${venv_path}"
    else
        echo "Using existing Blacklight GPU Python environment at ${venv_path}"
    fi
    uv pip install --python "${python_bin}" torch numpy
    "${python_bin}" "${SCRIPT_DIR}/blacklight_gpu_teacher_train.py" \
        --source-list /dev/null \
        --initial-model /dev/null \
        --check-backend
}

run_prune() {
    local apply="0"
    local keep_checkpoints="3"
    local prune_model_copies="1"
    local total_bytes=0
    local delete_count=0
    local path=""
    local size=0
    local manifest_path=""
    local checkpoint_dir=""
    local checkpoint_prefix_name=""
    local keep_list=""
    local chosen_checkpoint=""
    local current_model=""
    local active_model=""
    local source_model=""
    local -a protected_paths=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)
                apply="0"
                shift
                ;;
            --apply)
                apply="1"
                shift
                ;;
            --keep-checkpoints)
                keep_checkpoints="$2"
                shift 2
                ;;
            --skip-model-copies)
                prune_model_copies="0"
                shift
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                echo "Unknown prune option: $1" >&2
                usage >&2
                exit 1
                ;;
        esac
    done

    if ! [[ "${keep_checkpoints}" =~ ^[0-9]+$ ]]; then
        echo "--keep-checkpoints must be an integer." >&2
        exit 1
    fi

    normalize_existing_path() {
        local input_path="$1"
        if [[ -z "${input_path}" || ! -e "${input_path}" ]]; then
            return 1
        fi
        printf '%s/%s\n' "$(cd "$(dirname "${input_path}")" && pwd)" "$(basename "${input_path}")"
    }

    add_protected_path() {
        local input_path="$1"
        local normalized=""
        if normalized="$(normalize_existing_path "${input_path}")"; then
            protected_paths+=( "${normalized}" )
        fi
    }

    path_is_protected() {
        local input_path="$1"
        local normalized=""
        local protected=""
        if ! normalized="$(normalize_existing_path "${input_path}")"; then
            return 1
        fi
        for protected in "${protected_paths[@]}"; do
            if [[ "${normalized}" == "${protected}" ]]; then
                return 0
            fi
        done
        return 1
    }

    prune_file() {
        local input_path="$1"
        local file_size=0

        [[ -f "${input_path}" ]] || return 0
        if path_is_protected "${input_path}"; then
            return 0
        fi

        file_size="$(blacklight_file_size "${input_path}")"
        total_bytes=$(( total_bytes + file_size ))
        delete_count=$(( delete_count + 1 ))

        if [[ "${apply}" == "1" ]]; then
            echo "deleted ${input_path} $(blacklight_human_bytes "${file_size}")"
            rm -f "${input_path}"
        else
            echo "would_delete ${input_path} $(blacklight_human_bytes "${file_size}")"
        fi
    }

    if blacklight_source_champion_registry >/dev/null 2>&1; then
        current_model="${CHAMPION_MODEL_ABS:-}"
        active_model="${CHAMPION_ACTIVE_MODEL_ABS:-}"
        source_model="${SOURCE_MODEL_ABS:-}"
        add_protected_path "${current_model}"
        add_protected_path "${active_model}"
        add_protected_path "${source_model}"
    fi
    add_protected_path "${BLACKLIGHT_CHAMPION_MODEL_PATH}"

    while IFS= read -r manifest_path; do
        [[ -n "${manifest_path}" ]] || continue
        blacklight_source_manifest "${manifest_path}" || continue
        add_protected_path "${MODEL_ABS:-}"
        add_protected_path "${CHOSEN_CHECKPOINT_ABS:-}"
    done < <(blacklight_manifests_newest_first)

    if [[ "${prune_model_copies}" == "1" ]]; then
        while IFS= read -r path; do
            prune_file "${path}"
        done < <(
            find "${REPO_ROOT}/var/blacklight_eval" "${REPO_ROOT}/var/blacklight_bench" \
                -type f -name model.txt -print 2>/dev/null || true
        )
    fi

    while IFS= read -r manifest_path; do
        [[ -n "${manifest_path}" ]] || continue
        blacklight_source_manifest "${manifest_path}" || continue
        [[ -n "${CHECKPOINT_PREFIX_ABS:-}" ]] || continue
        checkpoint_dir="$(dirname "${CHECKPOINT_PREFIX_ABS}")"
        checkpoint_prefix_name="$(basename "${CHECKPOINT_PREFIX_ABS}")"
        [[ -d "${checkpoint_dir}" ]] || continue
        keep_list="$(find "${checkpoint_dir}" -maxdepth 1 -type f -name "${checkpoint_prefix_name}_ep*.txt" -print 2>/dev/null | sort | tail -n "${keep_checkpoints}")"
        while IFS= read -r chosen_checkpoint; do
            [[ -n "${chosen_checkpoint}" ]] || continue
            add_protected_path "${chosen_checkpoint}"
        done <<< "${keep_list}"
        while IFS= read -r path; do
            [[ -n "${path}" ]] || continue
            prune_file "${path}"
        done < <(find "${checkpoint_dir}" -maxdepth 1 -type f -name "${checkpoint_prefix_name}_ep*.txt" -print 2>/dev/null | sort)
    done < <(blacklight_manifests_newest_first)

    echo "Blacklight prune"
    echo "mode $([[ "${apply}" == "1" ]] && printf 'apply' || printf 'dry-run')"
    echo "files_matched ${delete_count}"
    echo "space_matched_bytes ${total_bytes}"
    echo "space_matched_human $(blacklight_human_bytes "${total_bytes}")"
    if [[ "${apply}" != "1" ]]; then
        echo "next_step ./scripts/blacklight.sh prune --apply"
    fi
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
    collect)
        run_collect "$@"
        ;;
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
    current)
        run_current "$@"
        ;;
    vast)
        exec "${SCRIPT_DIR}/blacklight_vast.sh" "$@"
        ;;
    champion)
        run_champion "$@"
        ;;
    gpu-setup)
        if [[ $# -gt 0 ]]; then
            echo "gpu-setup does not take options." >&2
            usage >&2
            exit 1
        fi
        run_gpu_setup
        ;;
    prune)
        run_prune "$@"
        ;;
    sweep)
        run_sweep "$@"
        ;;
    pipeline)
        run_pipeline "$@"
        ;;
    dashboard)
        run_dashboard "$@"
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
