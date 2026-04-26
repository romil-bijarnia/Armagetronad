#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ID="smoketest_$(date +%Y%m%d-%H%M%S)"

export ARMAGETRON_SELFPLAY_RUN_NAME="${RUN_ID}"
export ARMAGETRON_SELFPLAY_DURATION_SECONDS="${ARMAGETRON_SELFPLAY_DURATION_SECONDS:-60}"
export ARMAGETRON_SELFPLAY_LIMIT_ROUNDS="${ARMAGETRON_SELFPLAY_LIMIT_ROUNDS:-12}"
export ARMAGETRON_SELFPLAY_SAVE_EVERY="${ARMAGETRON_SELFPLAY_SAVE_EVERY:-1000}"
export ARMAGETRON_SELFPLAY_CHECKPOINT_EVERY="${ARMAGETRON_SELFPLAY_CHECKPOINT_EVERY:-0}"
export ARMAGETRON_SELFPLAY_PARALLEL_WORKERS="${ARMAGETRON_SELFPLAY_PARALLEL_WORKERS:-1}"
export ARMAGETRON_SELFPLAY_SYNC_SECONDS="${ARMAGETRON_SELFPLAY_SYNC_SECONDS:-20}"
export ARMAGETRON_SELFPLAY_PROFILE="${ARMAGETRON_SELFPLAY_PROFILE:-teacher}"

"${REPO_ROOT}/scripts/train_neural_ai_parallel.sh"

MANIFEST_PATH="${REPO_ROOT}/var/blacklight_runs/${RUN_ID}/run_manifest.env"
if [[ ! -f "${MANIFEST_PATH}" ]]; then
    echo "Smoke test failed: manifest not found at ${MANIFEST_PATH}" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${MANIFEST_PATH}"

if [[ ! -s "${MODEL_ABS}" ]]; then
    echo "Smoke test failed: model file missing or empty at ${MODEL_ABS}" >&2
    exit 1
fi

if [[ ! -s "${SOURCE_LIST_ABS}" ]]; then
    echo "Smoke test failed: source list missing or empty at ${SOURCE_LIST_ABS}" >&2
    exit 1
fi

FIRST_SOURCE_REL="$(sed -n '1p' "${SOURCE_LIST_ABS}")"
FIRST_SOURCE_ABS="${REPO_ROOT}/var/${FIRST_SOURCE_REL}"
if [[ ! -s "${FIRST_SOURCE_ABS}" ]]; then
    echo "Smoke test failed: teacher example log missing or empty at ${FIRST_SOURCE_ABS}" >&2
    exit 1
fi

if [[ ! -s "${METRICS_SUMMARY_ABS}" ]]; then
    echo "Smoke test failed: metrics summary missing or empty at ${METRICS_SUMMARY_ABS}" >&2
    exit 1
fi

MODEL_STATS="$(sed -n '3p' "${MODEL_ABS}")"
MODEL_EPISODES="$(printf '%s\n' "${MODEL_STATS}" | awk '{ print $2 }')"
MODEL_UPDATES="$(printf '%s\n' "${MODEL_STATS}" | awk '{ print $3 }')"
METRIC_EPISODES="$(awk '$1 == "episodes" { print $2 }' "${METRICS_SUMMARY_ABS}")"
AVERAGE_STEPS="$(awk '$1 == "average_steps" { print $2 }' "${METRICS_SUMMARY_ABS}")"
AVERAGE_POLICY_LOSS="$(awk '$1 == "average_policy_loss" { print $2 }' "${METRICS_SUMMARY_ABS}")"

if [[ -z "${MODEL_EPISODES}" || -z "${MODEL_UPDATES}" || "${MODEL_EPISODES}" == "0" || "${MODEL_UPDATES}" == "0" ]]; then
    echo "Smoke test failed: model file did not record training progress." >&2
    echo "Model stats line: ${MODEL_STATS}" >&2
    exit 1
fi

if [[ -z "${METRIC_EPISODES}" || "${METRIC_EPISODES}" == "0" ]]; then
    echo "Smoke test failed: metrics summary did not record any episodes." >&2
    exit 1
fi

if [[ -z "${AVERAGE_POLICY_LOSS}" || "${AVERAGE_POLICY_LOSS}" == "0" || "${AVERAGE_POLICY_LOSS}" == "0.000000" ]]; then
    echo "Smoke test failed: teacher training did not report a policy loss." >&2
    echo "Average steps: ${AVERAGE_STEPS}" >&2
    echo "Average policy loss: ${AVERAGE_POLICY_LOSS}" >&2
    exit 1
fi

if [[ "${ARMAGETRON_SELFPLAY_CHECKPOINT_EVERY}" =~ ^[0-9]+$ ]] && (( ARMAGETRON_SELFPLAY_CHECKPOINT_EVERY > 0 )); then
    LATEST_CHECKPOINT_FILE="${CHECKPOINT_PREFIX_ABS}_latest.txt"
    if [[ ! -s "${LATEST_CHECKPOINT_FILE}" ]]; then
        echo "Smoke test failed: checkpoint manifest missing or empty at ${LATEST_CHECKPOINT_FILE}" >&2
        exit 1
    fi

    CHECKPOINT_REL="$(sed -n '1p' "${LATEST_CHECKPOINT_FILE}")"
    CHECKPOINT_ABS="${REPO_ROOT}/var/${CHECKPOINT_REL}"
    if [[ ! -s "${CHECKPOINT_ABS}" ]]; then
        echo "Smoke test failed: checkpoint file missing or empty at ${CHECKPOINT_ABS}" >&2
        exit 1
    fi
else
    CHECKPOINT_ABS="disabled"
fi

echo "Smoke test passed."
echo "Manifest: ${MANIFEST_PATH}"
echo "Model episodes: ${MODEL_EPISODES}"
echo "Model updates: ${MODEL_UPDATES}"
echo "Metrics episodes: ${METRIC_EPISODES}"
echo "Teacher log: ${FIRST_SOURCE_ABS}"
echo "Checkpoint: ${CHECKPOINT_ABS}"
