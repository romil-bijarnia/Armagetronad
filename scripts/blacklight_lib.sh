#!/usr/bin/env bash

BLACKLIGHT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLACKLIGHT_REPO_ROOT="$(cd "${BLACKLIGHT_SCRIPT_DIR}/.." && pwd)"
BLACKLIGHT_LATEST_MANIFEST_PATH="${BLACKLIGHT_REPO_ROOT}/var/blacklight_runs/latest_run.env"
BLACKLIGHT_CHAMPION_DIR="${BLACKLIGHT_REPO_ROOT}/var/blacklight_champions"
BLACKLIGHT_CHAMPION_ENV_PATH="${BLACKLIGHT_CHAMPION_DIR}/current.env"
BLACKLIGHT_BENCH_ROOT="${BLACKLIGHT_REPO_ROOT}/var/blacklight_bench"
BLACKLIGHT_SWEEP_ROOT="${BLACKLIGHT_REPO_ROOT}/var/blacklight_sweeps"

blacklight_find_server_bin() {
    local preferred_bin="${1:-}"
    local candidates=(
        "${preferred_bin}"
        "${BLACKLIGHT_REPO_ROOT}/armagetronad-dedicated"
        "${BLACKLIGHT_REPO_ROOT}/src/armagetronad_main"
        "${BLACKLIGHT_REPO_ROOT}/src/armagetronad-dedicated"
        "/Applications/Armagetron Experimental.app/Contents/MacOS/armagetronad-dedicated"
        "/Applications/Armagetron Advanced.app/Contents/MacOS/armagetronad-dedicated"
    )
    local candidate

    for candidate in "${candidates[@]}"; do
        if [[ -n "${candidate}" && -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    return 1
}

blacklight_binary_rpaths() {
    local bin_path="$1"

    if [[ "$(uname -s)" != "Darwin" ]] || ! command -v otool >/dev/null 2>&1; then
        return 0
    fi

    otool -l "${bin_path}" 2>/dev/null | awk '
        $1 == "cmd" && $2 == "LC_RPATH" { want_path = 1; next }
        want_path && $1 == "path" { print $2; want_path = 0 }
    '
}

blacklight_resolve_macho_path() {
    local bin_path="$1"
    local install_name="$2"
    local binary_dir
    local rpath
    local candidate=""
    local fallback_candidate=""
    local suffix=""

    binary_dir="$(cd "$(dirname "${bin_path}")" && pwd)"

    case "${install_name}" in
        /*)
            printf '%s\n' "${install_name}"
            return 0
            ;;
        @loader_path/*)
            printf '%s/%s\n' "${binary_dir}" "${install_name#@loader_path/}"
            return 0
            ;;
        @executable_path/*)
            printf '%s/%s\n' "${binary_dir}" "${install_name#@executable_path/}"
            return 0
            ;;
        @rpath/*)
            suffix="${install_name#@rpath/}"
            while IFS= read -r rpath; do
                [[ -n "${rpath}" ]] || continue
                case "${rpath}" in
                    /*)
                        candidate="${rpath%/}/${suffix}"
                        ;;
                    @loader_path/*)
                        candidate="${binary_dir}/${rpath#@loader_path/}/${suffix}"
                        ;;
                    @executable_path/*)
                        candidate="${binary_dir}/${rpath#@executable_path/}/${suffix}"
                        ;;
                    *)
                        continue
                        ;;
                esac

                if [[ -e "${candidate}" ]]; then
                    printf '%s\n' "${candidate}"
                    return 0
                fi

                if [[ -z "${fallback_candidate}" && -n "${candidate}" ]]; then
                    fallback_candidate="${candidate}"
                fi
            done < <(blacklight_binary_rpaths "${bin_path}")
            if [[ -n "${fallback_candidate}" ]]; then
                printf '%s\n' "${fallback_candidate}"
                return 1
            fi
            ;;
    esac

    printf '%s\n' "${install_name}"
    return 1
}

blacklight_missing_runtime_libs() {
    local bin_path="$1"
    local lib_path
    local resolved_path

    if [[ "$(uname -s)" != "Darwin" ]] || ! command -v otool >/dev/null 2>&1; then
        return 0
    fi

    while IFS= read -r lib_path; do
        [[ -n "${lib_path}" ]] || continue
        resolved_path="$(blacklight_resolve_macho_path "${bin_path}" "${lib_path}")"

        case "${resolved_path}" in
            /usr/lib/*|/System/Library/*)
                continue
                ;;
        esac

        if [[ ! -e "${resolved_path}" ]]; then
            printf '%s\n' "${resolved_path}"
        fi
    done < <(otool -L "${bin_path}" 2>/dev/null | awk 'NR > 1 { print $1 }')
}

blacklight_preflight_server_bin() {
    local bin_path="$1"
    local missing_libs=""
    local missing_lib

    while IFS= read -r missing_lib; do
        [[ -n "${missing_lib}" ]] || continue
        missing_libs="${missing_libs}${missing_lib}"$'\n'
    done < <(blacklight_missing_runtime_libs "${bin_path}")

    if [[ -z "${missing_libs}" ]]; then
        return 0
    fi

    {
        echo "The selected Armagetron binary cannot start because runtime libraries are missing:"
        printf '%s' "${missing_libs}" | sed 's/^/  - /'
        echo
        echo "This build was linked against external libraries under /opt/homebrew."
        echo "Recover by either restoring those runtime libraries on this Mac"
        echo "or rebuilding Armagetron against the toolchain currently installed."
    } >&2

    return 1
}

blacklight_require_latest_manifest() {
    if [[ ! -f "${BLACKLIGHT_LATEST_MANIFEST_PATH}" ]]; then
        echo "No Blacklight run manifest found at ${BLACKLIGHT_LATEST_MANIFEST_PATH}." >&2
        echo "Run ./scripts/blacklight.sh train first." >&2
        return 1
    fi
}

blacklight_source_latest_manifest() {
    blacklight_require_latest_manifest || return 1
    # shellcheck disable=SC1090
    source "${BLACKLIGHT_LATEST_MANIFEST_PATH}"
}

blacklight_latest_candidate_model() {
    blacklight_source_latest_manifest || return 1

    if [[ -z "${MODEL_ABS:-}" || ! -f "${MODEL_ABS}" ]]; then
        echo "Latest manifest does not point to a candidate model." >&2
        return 1
    fi

    printf '%s\n' "${MODEL_ABS}"
}

blacklight_latest_reference_model() {
    blacklight_source_latest_manifest || return 1

    local latest_checkpoint_file="${CHECKPOINT_PREFIX_ABS:-}_latest.txt"
    if [[ ! -f "${latest_checkpoint_file}" ]]; then
        return 1
    fi

    local checkpoint_rel
    checkpoint_rel="$(sed -n '1p' "${latest_checkpoint_file}")"
    if [[ -z "${checkpoint_rel}" || ! -f "${BLACKLIGHT_REPO_ROOT}/var/${checkpoint_rel}" ]]; then
        return 1
    fi

    printf '%s\n' "${BLACKLIGHT_REPO_ROOT}/var/${checkpoint_rel}"
}

blacklight_read_summary_value() {
    local summary_path="$1"
    local key="$2"
    awk -v wanted="${key}" '$1 == wanted { print $2 }' "${summary_path}"
}

blacklight_latest_eval_report() {
    find "${BLACKLIGHT_REPO_ROOT}/var/blacklight_eval" -name report.txt -print 2>/dev/null | sort | tail -n 1 || true
}

blacklight_source_progress() {
    if [[ -n "${PROGRESS_ABS:-}" && -f "${PROGRESS_ABS}" ]]; then
        # shellcheck disable=SC1090
        source "${PROGRESS_ABS}"
    fi
}

blacklight_model_stats_line() {
    local model_path="$1"
    if [[ -f "${model_path}" ]]; then
        sed -n '3p' "${model_path}"
    fi
}

blacklight_model_stat_field() {
    local model_path="$1"
    local field_index="$2"
    blacklight_model_stats_line "${model_path}" | awk -v index="${field_index}" '{ print $index }'
}

blacklight_model_episodes() {
    local model_path="$1"
    local episodes
    episodes="$(blacklight_model_stat_field "${model_path}" 2)"
    printf '%s\n' "${episodes:-0}"
}

blacklight_model_updates() {
    local model_path="$1"
    local updates
    updates="$(blacklight_model_stat_field "${model_path}" 3)"
    printf '%s\n' "${updates:-0}"
}

blacklight_format_duration() {
    local seconds="${1:-0}"
    local sign=""

    if [[ -z "${seconds}" || ! "${seconds}" =~ ^-?[0-9]+$ ]]; then
        printf 'n/a\n'
        return 0
    fi

    if (( seconds < 0 )); then
        sign="-"
        seconds=$(( -seconds ))
    fi

    printf '%s%02d:%02d:%02d\n' \
        "${sign}" \
        $(( seconds / 3600 )) \
        $(( (seconds % 3600) / 60 )) \
        $(( seconds % 60 ))
}

blacklight_human_bytes() {
    local bytes="${1:-0}"
    local -a units=(B KB MB GB TB)
    local unit_index=0
    local whole
    local remainder=0
    local decimal=0

    if [[ -z "${bytes}" || ! "${bytes}" =~ ^[0-9]+$ ]]; then
        printf '0 B\n'
        return 0
    fi

    whole="${bytes}"
    while (( whole >= 1024 && unit_index < ${#units[@]} - 1 )); do
        remainder=$(( whole % 1024 ))
        whole=$(( whole / 1024 ))
        unit_index=$(( unit_index + 1 ))
    done

    if (( unit_index == 0 )); then
        printf '%s %s\n' "${whole}" "${units[${unit_index}]}"
        return 0
    fi

    decimal=$(( (remainder * 10 + 512) / 1024 ))
    if (( decimal == 10 )); then
        whole=$(( whole + 1 ))
        decimal=0
    fi

    printf '%s.%s %s\n' "${whole}" "${decimal}" "${units[${unit_index}]}"
}

blacklight_file_size() {
    local path="$1"

    if [[ ! -f "${path}" ]]; then
        printf '0\n'
        return 0
    fi

    wc -c < "${path}" | tr -d '[:space:]'
    printf '\n'
}

blacklight_sum_file_sizes() {
    local total=0
    local path
    local size

    for path in "$@"; do
        if [[ -f "${path}" ]]; then
            size="$(wc -c < "${path}")"
            size="${size//[[:space:]]/}"
            total=$(( total + size ))
        fi
    done

    printf '%s\n' "${total}"
}

blacklight_count_running_pids() {
    local count=0
    local pid_list
    local pid

    for pid_list in "$@"; do
        for pid in ${pid_list}; do
            if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
                count=$(( count + 1 ))
            fi
        done
    done

    printf '%s\n' "${count}"
}

blacklight_progress_percent() {
    local current="${1:-0}"
    local total="${2:-0}"

    if [[ -z "${current}" || ! "${current}" =~ ^-?[0-9]+$ ]]; then
        current=0
    fi
    if [[ -z "${total}" || ! "${total}" =~ ^[0-9]+$ ]] || (( total <= 0 )); then
        printf '0\n'
        return 0
    fi

    if (( current < 0 )); then
        current=0
    elif (( current > total )); then
        current="${total}"
    fi

    printf '%s\n' $(( (current * 100 + total / 2) / total ))
}

blacklight_render_progress_bar() {
    local current="${1:-0}"
    local total="${2:-0}"
    local width="${3:-24}"
    local fill_char="${4:-#}"
    local empty_char="${5:--}"
    local filled=0
    local index

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

    printf '['
    for (( index = 0; index < width; ++index )); do
        if (( index < filled )); then
            printf '%s' "${fill_char}"
        else
            printf '%s' "${empty_char}"
        fi
    done
    printf ']'
}

blacklight_render_activity_bar() {
    local tick="${1:-0}"
    local width="${2:-24}"
    local index
    local position

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

    printf '['
    for (( index = 0; index < width; ++index )); do
        if (( index < position )); then
            printf '='
        elif (( index == position )); then
            printf '>'
        else
            printf '-'
        fi
    done
    printf ']'
}

blacklight_train_profile_default() {
    printf 'classic_bootstrap\n'
}

blacklight_train_profile_exists() {
    case "${1:-}" in
        classic_bootstrap|classic_hardening|mixed_league|none)
            return 0
            ;;
    esac

    return 1
}

blacklight_train_profile_value() {
    local profile="${1:-}"
    local key="${2:-}"

    case "${profile}:${key}" in
        classic_bootstrap:training_mode) printf 'parallel\n' ;;
        classic_bootstrap:bot_count) printf '1\n' ;;
        classic_bootstrap:exploration) printf '0.16\n' ;;
        classic_bootstrap:train_epochs) printf '6\n' ;;
        classic_bootstrap:record_stride) printf '32\n' ;;
        classic_bootstrap:save_every) printf '100\n' ;;
        classic_bootstrap:checkpoint_every) printf '100\n' ;;
        classic_bootstrap:parallel_workers) printf '8\n' ;;
        classic_bootstrap:sync_seconds) printf '300\n' ;;
        classic_bootstrap:duration_seconds) printf '21600\n' ;;
        classic_bootstrap:policy_pool_size) printf '0\n' ;;
        classic_bootstrap:policy_historic_prob) printf '0\n' ;;
        classic_bootstrap:min_players) printf '8\n' ;;
        classic_bootstrap:teams_min) printf '8\n' ;;
        classic_bootstrap:teams_max) printf '8\n' ;;
        classic_hardening:training_mode) printf 'parallel\n' ;;
        classic_hardening:bot_count) printf '2\n' ;;
        classic_hardening:exploration) printf '0.08\n' ;;
        classic_hardening:train_epochs) printf '6\n' ;;
        classic_hardening:record_stride) printf '32\n' ;;
        classic_hardening:parallel_workers) printf '8\n' ;;
        classic_hardening:sync_seconds) printf '600\n' ;;
        classic_hardening:duration_seconds) printf '28800\n' ;;
        classic_hardening:policy_pool_size) printf '8\n' ;;
        classic_hardening:policy_snapshot_every) printf '40\n' ;;
        classic_hardening:policy_snapshot_warmup) printf '80\n' ;;
        classic_hardening:policy_historic_prob) printf '0.25\n' ;;
        classic_hardening:min_players) printf '8\n' ;;
        classic_hardening:teams_min) printf '8\n' ;;
        classic_hardening:teams_max) printf '8\n' ;;
        mixed_league:training_mode) printf 'parallel\n' ;;
        mixed_league:bot_count) printf '4\n' ;;
        mixed_league:exploration) printf '0.04\n' ;;
        mixed_league:train_epochs) printf '4\n' ;;
        mixed_league:record_stride) printf '32\n' ;;
        mixed_league:parallel_workers) printf '8\n' ;;
        mixed_league:sync_seconds) printf '600\n' ;;
        mixed_league:duration_seconds) printf '14400\n' ;;
        mixed_league:policy_pool_size) printf '8\n' ;;
        mixed_league:policy_snapshot_every) printf '40\n' ;;
        mixed_league:policy_snapshot_warmup) printf '80\n' ;;
        mixed_league:policy_historic_prob) printf '0.35\n' ;;
        mixed_league:min_players) printf '8\n' ;;
        mixed_league:teams_min) printf '8\n' ;;
        mixed_league:teams_max) printf '8\n' ;;
    esac
}

blacklight_bench_suite_exists() {
    case "${1:-}" in
        classic_primary|mixed_secondary)
            return 0
            ;;
    esac

    return 1
}

blacklight_bench_suite_value() {
    local suite="${1:-}"
    local key="${2:-}"

    case "${suite}:${key}" in
        classic_primary:sessions) printf '20\n' ;;
        classic_primary:limit_rounds) printf '100\n' ;;
        classic_primary:duration_seconds) printf '300\n' ;;
        classic_primary:bot_count) printf '1\n' ;;
        classic_primary:min_players) printf '8\n' ;;
        classic_primary:teams_min) printf '8\n' ;;
        classic_primary:teams_max) printf '8\n' ;;
        mixed_secondary:sessions) printf '10\n' ;;
        mixed_secondary:limit_rounds) printf '100\n' ;;
        mixed_secondary:duration_seconds) printf '300\n' ;;
        mixed_secondary:bot_count) printf '2\n' ;;
        mixed_secondary:min_players) printf '8\n' ;;
        mixed_secondary:teams_min) printf '8\n' ;;
        mixed_secondary:teams_max) printf '8\n' ;;
    esac
}

blacklight_env_get() {
    local env_path="$1"
    local key="$2"
    awk -F= -v wanted="${key}" '$1 == wanted { sub(/^[^=]*=/, ""); print; exit }' "${env_path}"
}

blacklight_env_set() {
    local env_path="$1"
    local key="$2"
    local value="$3"
    local tmp_path

    mkdir -p "$(dirname "${env_path}")"
    tmp_path="$(mktemp)"

    if [[ -f "${env_path}" ]]; then
        awk -F= -v wanted="${key}" -v replacement="${value}" '
            BEGIN { updated = 0 }
            $1 == wanted {
                print wanted "=" replacement
                updated = 1
                next
            }
            { print }
            END {
                if (!updated) {
                    print wanted "=" replacement
                }
            }
        ' "${env_path}" > "${tmp_path}"
    else
        printf '%s=%s\n' "${key}" "${value}" > "${tmp_path}"
    fi

    mv "${tmp_path}" "${env_path}"
}

blacklight_report_value() {
    local report_path="$1"
    local key="$2"
    awk -v wanted="${key}" '$1 == wanted { print $2; exit }' "${report_path}"
}

blacklight_source_manifest() {
    local manifest_path="$1"
    if [[ ! -f "${manifest_path}" ]]; then
        return 1
    fi

    # shellcheck disable=SC1090
    source "${manifest_path}"
}

blacklight_source_champion_registry() {
    if [[ ! -f "${BLACKLIGHT_CHAMPION_ENV_PATH}" ]]; then
        return 1
    fi

    # shellcheck disable=SC1090
    source "${BLACKLIGHT_CHAMPION_ENV_PATH}"
}

blacklight_current_champion_model() {
    if ! blacklight_source_champion_registry; then
        return 1
    fi

    if [[ -n "${CHAMPION_MODEL_ABS:-}" && -f "${CHAMPION_MODEL_ABS}" ]]; then
        printf '%s\n' "${CHAMPION_MODEL_ABS}"
        return 0
    fi

    return 1
}

blacklight_path_abs() {
    local input_path="$1"

    case "${input_path}" in
        /*)
            printf '%s\n' "${input_path}"
            ;;
        *)
            printf '%s/%s\n' "$(pwd)" "${input_path}"
            ;;
    esac
}

blacklight_resolve_run_dir() {
    local input="${1:-}"
    local abs_input

    if [[ -z "${input}" ]]; then
        return 1
    fi

    if [[ -d "${input}" && -f "${input}/run_manifest.env" ]]; then
        abs_input="$(cd "${input}" && pwd)"
        printf '%s\n' "${abs_input}"
        return 0
    fi

    if [[ -f "${input}" && "$(basename "${input}")" == "run_manifest.env" ]]; then
        abs_input="$(cd "$(dirname "${input}")" && pwd)"
        printf '%s\n' "${abs_input}"
        return 0
    fi

    if [[ -d "${BLACKLIGHT_REPO_ROOT}/var/blacklight_runs/${input}" && -f "${BLACKLIGHT_REPO_ROOT}/var/blacklight_runs/${input}/run_manifest.env" ]]; then
        printf '%s\n' "${BLACKLIGHT_REPO_ROOT}/var/blacklight_runs/${input}"
        return 0
    fi

    return 1
}

blacklight_manifest_path_for_input() {
    local input="${1:-}"
    local run_dir=""

    if run_dir="$(blacklight_resolve_run_dir "${input}")"; then
        printf '%s/run_manifest.env\n' "${run_dir}"
        return 0
    fi

    if [[ -f "${input}" && "$(basename "${input}")" == "run_manifest.env" ]]; then
        printf '%s\n' "$(blacklight_path_abs "${input}")"
        return 0
    fi

    return 1
}

blacklight_resolve_model_input() {
    local input="${1:-}"
    local prefer_chosen="${2:-1}"
    local manifest_path=""

    if [[ -z "${input}" ]]; then
        blacklight_latest_candidate_model
        return $?
    fi

    if [[ "${input}" == "champion" ]]; then
        blacklight_current_champion_model
        return $?
    fi

    if [[ -f "${input}" && "$(basename "${input}")" != "run_manifest.env" ]]; then
        printf '%s\n' "$(blacklight_path_abs "${input}")"
        return 0
    fi

    if manifest_path="$(blacklight_manifest_path_for_input "${input}")"; then
        blacklight_source_manifest "${manifest_path}" || return 1
        if [[ "${prefer_chosen}" == "1" && -n "${CHOSEN_CHECKPOINT_ABS:-}" && -f "${CHOSEN_CHECKPOINT_ABS}" ]]; then
            printf '%s\n' "${CHOSEN_CHECKPOINT_ABS}"
            return 0
        fi
        if [[ -n "${MODEL_ABS:-}" && -f "${MODEL_ABS}" ]]; then
            printf '%s\n' "${MODEL_ABS}"
            return 0
        fi
    fi

    return 1
}

blacklight_checkpoint_candidates_for_input() {
    local input="$1"
    local limit="${2:-3}"
    local manifest_path=""

    if ! manifest_path="$(blacklight_manifest_path_for_input "${input}")"; then
        return 1
    fi

    blacklight_source_manifest "${manifest_path}" || return 1

    {
        if [[ -n "${MODEL_ABS:-}" && -f "${MODEL_ABS}" ]]; then
            printf '%s\n' "${MODEL_ABS}"
        fi
        if [[ -n "${CHECKPOINT_PREFIX_ABS:-}" ]]; then
            find "$(dirname "${CHECKPOINT_PREFIX_ABS}")" -maxdepth 1 -type f -name "$(basename "${CHECKPOINT_PREFIX_ABS}")_ep*.txt" -print 2>/dev/null | sort | tail -n "${limit}"
        fi
    } | awk 'NF > 0 && !seen[$0]++'
}

blacklight_read_metric_window() {
    local metrics_path="$1"
    local window="$2"
    local column="$3"

    if [[ ! -f "${metrics_path}" ]]; then
        printf 'n/a\n'
        return 0
    fi

    awk -F, -v window="${window}" -v column="${column}" '
        NR == 1 { next }
        {
            values[count % window] = $column
            count++
        }
        END {
            if (count == 0) {
                print "n/a"
                exit
            }
            limit = count < window ? count : window
            total = 0
            for (i = 0; i < limit; ++i) {
                total += values[i]
            }
            printf "%.6f\n", total / limit
        }
    ' "${metrics_path}"
}

blacklight_read_sync_metric() {
    local events_path="$1"
    local mode="${2:-avg}"

    if [[ ! -f "${events_path}" ]]; then
        printf 'n/a\n'
        return 0
    fi

    awk -v mode="${mode}" '
        /sync complete/ {
            for (i = 1; i <= NF; ++i) {
                if ($i ~ /^duration=/) {
                    split($i, parts, "=")
                    value = parts[2] + 0
                    total += value
                    count++
                    last = value
                }
            }
        }
        END {
            if (count == 0) {
                print "n/a"
            } else if (mode == "last") {
                print last
            } else {
                printf "%.2f\n", total / count
            }
        }
    ' "${events_path}"
}

blacklight_current_experience_growth() {
    local current_bytes="${1:-0}"
    local last_sync_bytes="${2:-0}"

    if [[ -z "${current_bytes}" || ! "${current_bytes}" =~ ^[0-9]+$ ]]; then
        current_bytes=0
    fi
    if [[ -z "${last_sync_bytes}" || ! "${last_sync_bytes}" =~ ^[0-9]+$ ]]; then
        last_sync_bytes=0
    fi

    if (( current_bytes < last_sync_bytes )); then
        printf '0\n'
    else
        printf '%s\n' $(( current_bytes - last_sync_bytes ))
    fi
}

blacklight_format_source_progress() {
    local state_path="$1"
    local source_list_path="$2"

    if [[ ! -f "${state_path}" || ! -f "${source_list_path}" ]]; then
        return 0
    fi

    awk '
        FNR == NR {
            if ($1 == "source") {
                offsets[$2] = $3
            } else if ($1 == "source_episodes") {
                episodes[$2] = $3
            }
            next
        }
        NF > 0 {
            sources[++count] = $1
        }
        END {
            for (i = 1; i <= count; ++i) {
                path = sources[i]
                printf "%s|%s|%s\n", path, (path in episodes ? episodes[path] : 0), (path in offsets ? offsets[path] : 0)
            }
        }
    ' "${state_path}" "${source_list_path}"
}

blacklight_float_is_better() {
    local win_a="$1"
    local dist_a="$2"
    local win_b="$3"
    local dist_b="$4"

    awk -v win_a="${win_a}" -v dist_a="${dist_a}" -v win_b="${win_b}" -v dist_b="${dist_b}" '
        BEGIN {
            if (win_a > win_b + 0.0000005) {
                exit 0
            }
            if (win_b > win_a + 0.0000005) {
                exit 1
            }
            if (dist_a > dist_b + 0.0000005) {
                exit 0
            }
            exit 1
        }
    '
}

blacklight_pick_best_report() {
    local best_report=""
    local best_win=""
    local best_dist=""
    local report_path
    local win_rate
    local distance

    for report_path in "$@"; do
        [[ -f "${report_path}" ]] || continue
        win_rate="$(blacklight_report_value "${report_path}" candidate_mean_win_rate)"
        distance="$(blacklight_report_value "${report_path}" candidate_mean_distance)"
        if [[ -z "${best_report}" ]]; then
            best_report="${report_path}"
            best_win="${win_rate}"
            best_dist="${distance}"
            continue
        fi
        if blacklight_float_is_better "${win_rate}" "${distance}" "${best_win}" "${best_dist}"; then
            best_report="${report_path}"
            best_win="${win_rate}"
            best_dist="${distance}"
        fi
    done

    [[ -n "${best_report}" ]] || return 1
    printf '%s\n' "${best_report}"
}

blacklight_promote_threshold_passes() {
    local candidate_win="$1"
    local reference_win="$2"
    local candidate_distance="$3"
    local reference_distance="$4"

    awk \
        -v candidate_win="${candidate_win}" \
        -v reference_win="${reference_win}" \
        -v candidate_distance="${candidate_distance}" \
        -v reference_distance="${reference_distance}" '
        BEGIN {
            if (candidate_win + 0 < reference_win + 0.02 - 0.0000005) {
                exit 1
            }
            if (reference_distance > 0 && candidate_distance < reference_distance * 0.95 - 0.0000005) {
                exit 1
            }
            exit 0
        }
    '
}

blacklight_mixed_veto_passes() {
    local candidate_win="$1"
    local reference_win="$2"
    local candidate_distance="$3"
    local reference_distance="$4"

    awk \
        -v candidate_win="${candidate_win}" \
        -v reference_win="${reference_win}" \
        -v candidate_distance="${candidate_distance}" \
        -v reference_distance="${reference_distance}" '
        BEGIN {
            if (candidate_win < reference_win - 0.03 - 0.0000005) {
                exit 1
            }
            if (reference_distance > 0 && candidate_distance < reference_distance * 0.90 - 0.0000005) {
                exit 1
            }
            exit 0
        }
    '
}
