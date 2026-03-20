#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/blacklight_lib.sh"
REPO_ROOT="${BLACKLIGHT_REPO_ROOT}"

EVAL_DURATION_SECONDS="${ARMAGETRON_EVAL_DURATION_SECONDS:-30}"
EVAL_LIMIT_ROUNDS="${ARMAGETRON_EVAL_LIMIT_ROUNDS:-12}"
BASE_CFG_REL="${ARMAGETRON_EVAL_BASE_CFG_REL:-examples/trained_ai_selfplay.cfg}"
EVAL_ID="${ARMAGETRON_EVAL_ID:-$(date +%Y%m%d-%H%M%S)}"
EVAL_ROOT_REL="blacklight_eval/${EVAL_ID}"
EVAL_ROOT_ABS="${REPO_ROOT}/var/${EVAL_ROOT_REL}"

mkdir -p "${EVAL_ROOT_ABS}"

if ! BIN_PATH="$(blacklight_find_server_bin "${ARMAGETRON_SELFPLAY_BIN:-}")"; then
    echo "Could not find a dedicated-capable server binary for evaluation." >&2
    exit 1
fi
if ! blacklight_preflight_server_bin "${BIN_PATH}"; then
    exit 1
fi

copy_model_into_var() {
    local source_path="$1"
    local target_rel="$2"
    local target_abs="${REPO_ROOT}/var/${target_rel}"
    mkdir -p "$(dirname "${target_abs}")"
    cp "${source_path}" "${target_abs}"
    printf '%s\n' "${target_rel}"
}

resolve_default_candidate() {
    if ! blacklight_latest_candidate_model; then
        exit 1
    fi
}

resolve_default_reference() {
    blacklight_latest_reference_model
}

run_eval_session() {
    local label="$1"
    local model_abs="${2:-}"
    local enable_trained="$3"

    local session_rel="${EVAL_ROOT_REL}/${label}"
    local session_abs="${REPO_ROOT}/var/${session_rel}"
    local cfg_rel="generated_eval_${EVAL_ID}_${label}.cfg"
    local cfg_path="${REPO_ROOT}/config/${cfg_rel}"
    local eval_metrics_rel="${session_rel}/ai_eval_metrics.csv"
    local eval_metrics_summary="${REPO_ROOT}/var/${eval_metrics_rel}.latest"
    local blacklight_metrics_rel="${session_rel}/blacklight_metrics.csv"
    local server_log="${session_abs}/server.log"
    local model_rel=""

    mkdir -p "${session_abs}"

    if [[ -n "${model_abs}" ]]; then
        model_rel="$(copy_model_into_var "${model_abs}" "${session_rel}/model.txt")"
    fi

    cat > "${cfg_path}" <<EOF
SINCLUDE ${BASE_CFG_REL}
AI_TRAINED_ENABLE ${enable_trained}
AI_TRAINED_BOT_COUNT $([[ "${enable_trained}" == "1" ]] && printf '%s' "-1" || printf '%s' "0")
AI_TRAINED_LEARN 0
AI_TRAINED_RECORD 0
AI_TRAINED_METRICS_FILE ${blacklight_metrics_rel}
AI_TRAINED_POLICY_POOL_SIZE 0
AI_TRAINED_POLICY_HISTORIC_PROB 0
AI_EVAL_METRICS_FILE ${eval_metrics_rel}
WIN_ZONE_RANDOMNESS 0
LIMIT_ROUNDS ${EVAL_LIMIT_ROUNDS}
EOF

    if [[ -n "${model_rel}" ]]; then
        printf 'AI_TRAINED_MODEL_FILE %s\n' "${model_rel}" >> "${cfg_path}"
    fi

    "${BIN_PATH}" \
        --datadir "${REPO_ROOT}" \
        --configdir "${REPO_ROOT}/config" \
        --userdatadir "${REPO_ROOT}/var" \
        --vardir "${REPO_ROOT}/var" \
        --extraconfig "${cfg_rel}" > "${server_log}" 2>&1 &
    local server_pid=$!
    sleep "${EVAL_DURATION_SECONDS}"
    kill -TERM "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" || true

    if [[ ! -f "${eval_metrics_summary}" ]]; then
        echo "Evaluation session ${label} did not produce ${eval_metrics_summary}" >&2
        return 1
    fi

    printf '%s\n' "${eval_metrics_summary}"
}

read_summary_value() {
    local summary_path="$1"
    local key="$2"
    blacklight_read_summary_value "${summary_path}" "${key}"
}

CANDIDATE_SOURCE="${1:-}"
REFERENCE_SOURCE="${2:-}"

if [[ -z "${CANDIDATE_SOURCE}" ]]; then
    CANDIDATE_SOURCE="$(resolve_default_candidate)"
fi

if [[ -z "${REFERENCE_SOURCE}" ]]; then
    REFERENCE_SOURCE="$(resolve_default_reference || true)"
fi

CANDIDATE_SUMMARY="$(run_eval_session candidate "${CANDIDATE_SOURCE}" 1)"
REFERENCE_SUMMARY=""
if [[ -n "${REFERENCE_SOURCE}" ]]; then
    REFERENCE_SUMMARY="$(run_eval_session reference "${REFERENCE_SOURCE}" 1)"
fi
CLASSIC_SUMMARY="$(run_eval_session classic "" 0)"

REPORT_PATH="${EVAL_ROOT_ABS}/report.txt"
{
    echo "Blacklight evaluation"
    echo "candidate_model ${CANDIDATE_SOURCE}"
    echo "duration_seconds ${EVAL_DURATION_SECONDS}"
    echo "limit_rounds ${EVAL_LIMIT_ROUNDS}"
    echo
    echo "candidate_kind $(read_summary_value "${CANDIDATE_SUMMARY}" kind)"
    echo "candidate_episodes $(read_summary_value "${CANDIDATE_SUMMARY}" episodes)"
    echo "candidate_win_rate $(read_summary_value "${CANDIDATE_SUMMARY}" win_rate)"
    echo "candidate_average_distance $(read_summary_value "${CANDIDATE_SUMMARY}" average_distance)"
    if [[ -n "${REFERENCE_SUMMARY}" ]]; then
        echo
        echo "reference_model ${REFERENCE_SOURCE}"
        echo "reference_kind $(read_summary_value "${REFERENCE_SUMMARY}" kind)"
        echo "reference_episodes $(read_summary_value "${REFERENCE_SUMMARY}" episodes)"
        echo "reference_win_rate $(read_summary_value "${REFERENCE_SUMMARY}" win_rate)"
        echo "reference_average_distance $(read_summary_value "${REFERENCE_SUMMARY}" average_distance)"
    fi
    echo
    echo "classic_kind $(read_summary_value "${CLASSIC_SUMMARY}" kind)"
    echo "classic_episodes $(read_summary_value "${CLASSIC_SUMMARY}" episodes)"
    echo "classic_win_rate $(read_summary_value "${CLASSIC_SUMMARY}" win_rate)"
    echo "classic_average_distance $(read_summary_value "${CLASSIC_SUMMARY}" average_distance)"
} > "${REPORT_PATH}"

cat "${REPORT_PATH}"
