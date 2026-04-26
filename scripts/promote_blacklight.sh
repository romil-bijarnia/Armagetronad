#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/blacklight_lib.sh"
REPO_ROOT="${BLACKLIGHT_REPO_ROOT}"

CANDIDATE_SOURCE=""

usage() {
    cat <<'EOF'
Usage:
  ./scripts/promote_blacklight.sh [--candidate MODEL_PATH_OR_RUN]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --candidate)
            CANDIDATE_SOURCE="$2"
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

if [[ -z "${CANDIDATE_SOURCE}" ]]; then
    CANDIDATE_SOURCE="latest"
fi

if ! CANDIDATE_MODEL="$(blacklight_resolve_model_input "${CANDIDATE_SOURCE}" 1)"; then
    echo "Could not resolve candidate model: ${CANDIDATE_SOURCE}" >&2
    exit 1
fi

CANDIDATE_MANIFEST="$(blacklight_manifest_path_for_input "${CANDIDATE_SOURCE}" || true)"
REFERENCE_MODEL="$(blacklight_current_champion_model || true)"
PROMOTION_STATUS="rejected"
PROMOTION_REASON="candidate did not clear the promotion thresholds"
PROMOTED_AT="$(date '+%Y-%m-%d %H:%M:%S')"

run_bench_capture() {
    local suite="$1"
    shift
    local output

    output="$("${SCRIPT_DIR}/benchmark_blacklight.sh" --suite "${suite}" --candidate "${CANDIDATE_SOURCE}" "$@")"
    printf '%s\n' "${output}" >&2
    printf '%s\n' "${output}" | awk '$1 == "report_path" { print $2; exit }'
}

CLASSIC_REPORT=""
MIXED_REPORT=""

if [[ -n "${REFERENCE_MODEL}" ]]; then
    CLASSIC_REPORT="$(run_bench_capture classic_primary --reference "${REFERENCE_MODEL}")"
    MIXED_REPORT="$(run_bench_capture mixed_secondary --reference "${REFERENCE_MODEL}")"
else
    CLASSIC_REPORT="$(run_bench_capture classic_primary)"
    MIXED_REPORT="$(run_bench_capture mixed_secondary)"
fi

[[ -n "${CLASSIC_REPORT}" && -f "${CLASSIC_REPORT}" ]] || { echo "Classic promotion benchmark did not produce a report." >&2; exit 1; }
[[ -n "${MIXED_REPORT}" && -f "${MIXED_REPORT}" ]] || { echo "Mixed promotion benchmark did not produce a report." >&2; exit 1; }

if [[ -z "${REFERENCE_MODEL}" ]]; then
    PROMOTION_STATUS="promoted"
    PROMOTION_REASON="no existing champion, seeding initial champion after benchmarks"
else
    CANDIDATE_CLASSIC_WIN="$(blacklight_report_value "${CLASSIC_REPORT}" candidate_mean_win_rate)"
    REFERENCE_CLASSIC_WIN="$(blacklight_report_value "${CLASSIC_REPORT}" reference_mean_win_rate)"
    CANDIDATE_CLASSIC_DISTANCE="$(blacklight_report_value "${CLASSIC_REPORT}" candidate_mean_distance)"
    REFERENCE_CLASSIC_DISTANCE="$(blacklight_report_value "${CLASSIC_REPORT}" reference_mean_distance)"
    CANDIDATE_MIXED_WIN="$(blacklight_report_value "${MIXED_REPORT}" candidate_mean_win_rate)"
    REFERENCE_MIXED_WIN="$(blacklight_report_value "${MIXED_REPORT}" reference_mean_win_rate)"
    CANDIDATE_MIXED_DISTANCE="$(blacklight_report_value "${MIXED_REPORT}" candidate_mean_distance)"
    REFERENCE_MIXED_DISTANCE="$(blacklight_report_value "${MIXED_REPORT}" reference_mean_distance)"

    if blacklight_promote_threshold_passes \
        "${CANDIDATE_CLASSIC_WIN}" \
        "${REFERENCE_CLASSIC_WIN}" \
        "${CANDIDATE_CLASSIC_DISTANCE}" \
        "${REFERENCE_CLASSIC_DISTANCE}" &&
        blacklight_mixed_veto_passes \
            "${CANDIDATE_MIXED_WIN}" \
            "${REFERENCE_MIXED_WIN}" \
            "${CANDIDATE_MIXED_DISTANCE}" \
            "${REFERENCE_MIXED_DISTANCE}"; then
        PROMOTION_STATUS="promoted"
        PROMOTION_REASON="candidate cleared classic and mixed champion gates"
    fi
fi

if [[ -n "${CANDIDATE_MANIFEST}" && -f "${CANDIDATE_MANIFEST}" ]]; then
    blacklight_env_set "${CANDIDATE_MANIFEST}" "BENCH_REPORT_ABS" "${CLASSIC_REPORT}"
    blacklight_env_set "${CANDIDATE_MANIFEST}" "CLASSIC_BENCH_REPORT_ABS" "${CLASSIC_REPORT}"
    blacklight_env_set "${CANDIDATE_MANIFEST}" "MIXED_BENCH_REPORT_ABS" "${MIXED_REPORT}"
    blacklight_env_set "${CANDIDATE_MANIFEST}" "PROMOTION_STATUS" "${PROMOTION_STATUS}"
    blacklight_env_set "${CANDIDATE_MANIFEST}" "PROMOTION_REASON" "${PROMOTION_REASON}"
    blacklight_env_set "${CANDIDATE_MANIFEST}" "CHOSEN_CHECKPOINT_ABS" "${CANDIDATE_MODEL}"
fi

if [[ "${PROMOTION_STATUS}" == "promoted" ]]; then
    CHAMPION_MODEL_ABS="$(blacklight_install_champion_model \
        "${CANDIDATE_MODEL}" \
        "${CANDIDATE_SOURCE}" \
        "${CLASSIC_REPORT}" \
        "${MIXED_REPORT}" \
        "${PROMOTION_STATUS}" \
        "${PROMOTED_AT}")"
fi

echo "Blacklight promotion"
echo "candidate_source ${CANDIDATE_SOURCE}"
echo "candidate_model ${CANDIDATE_MODEL}"
if [[ -n "${REFERENCE_MODEL}" ]]; then
    echo "reference_model ${REFERENCE_MODEL}"
fi
echo "classic_bench_report ${CLASSIC_REPORT}"
echo "mixed_bench_report ${MIXED_REPORT}"
echo "promotion_status ${PROMOTION_STATUS}"
echo "promotion_reason ${PROMOTION_REASON}"
if [[ "${PROMOTION_STATUS}" == "promoted" ]]; then
    echo "champion_model ${CHAMPION_MODEL_ABS}"
    echo "champion_active_model ${BLACKLIGHT_CHAMPION_MODEL_PATH}"
    echo "champion_registry ${BLACKLIGHT_CHAMPION_ENV_PATH}"
fi
