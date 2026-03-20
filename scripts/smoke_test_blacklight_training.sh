#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ID="smoketest_$(date +%Y%m%d-%H%M%S)"

export ARMAGETRON_SELFPLAY_RUN_NAME="${RUN_ID}"
export ARMAGETRON_SELFPLAY_DURATION_SECONDS="${ARMAGETRON_SELFPLAY_DURATION_SECONDS:-120}"
export ARMAGETRON_SELFPLAY_LIMIT_ROUNDS="${ARMAGETRON_SELFPLAY_LIMIT_ROUNDS:-12}"
export ARMAGETRON_SELFPLAY_SAVE_EVERY="${ARMAGETRON_SELFPLAY_SAVE_EVERY:-1}"
export ARMAGETRON_SELFPLAY_CHECKPOINT_EVERY="${ARMAGETRON_SELFPLAY_CHECKPOINT_EVERY:-1}"

"${REPO_ROOT}/scripts/train_neural_ai_selfplay.sh"

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

if [[ ! -s "${RECORD_ABS}" ]]; then
    echo "Smoke test failed: experience log missing or empty at ${RECORD_ABS}" >&2
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
AVERAGE_DISTANCE="$(awk '$1 == "average_distance" { print $2 }' "${METRICS_SUMMARY_ABS}")"

if [[ -z "${MODEL_EPISODES}" || -z "${MODEL_UPDATES}" || "${MODEL_EPISODES}" == "0" || "${MODEL_UPDATES}" == "0" ]]; then
    echo "Smoke test failed: model file did not record training progress." >&2
    echo "Model stats line: ${MODEL_STATS}" >&2
    exit 1
fi

if [[ -z "${METRIC_EPISODES}" || "${METRIC_EPISODES}" == "0" ]]; then
    echo "Smoke test failed: metrics summary did not record any episodes." >&2
    exit 1
fi

if [[ "${AVERAGE_STEPS:-0}" == "1.000000" && "${AVERAGE_DISTANCE:-1}" == "0.000000" ]]; then
    echo "Smoke test failed: training rounds collapsed into one-step zero-distance wins." >&2
    echo "Average steps: ${AVERAGE_STEPS}" >&2
    echo "Average distance: ${AVERAGE_DISTANCE}" >&2
    exit 1
fi

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

echo "Smoke test passed."
echo "Manifest: ${MANIFEST_PATH}"
echo "Model episodes: ${MODEL_EPISODES}"
echo "Model updates: ${MODEL_UPDATES}"
echo "Metrics episodes: ${METRIC_EPISODES}"
echo "Checkpoint: ${CHECKPOINT_ABS}"
