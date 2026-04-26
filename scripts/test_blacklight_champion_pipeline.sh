#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/blacklight_lib.sh"
REPO_ROOT="${BLACKLIGHT_REPO_ROOT}"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_file() {
    local path="$1"
    [[ -f "${path}" ]] || fail "Expected file: ${path}"
}

assert_nonempty_file() {
    local path="$1"
    [[ -s "${path}" ]] || fail "Expected non-empty file: ${path}"
}

assert_contains() {
    local path="$1"
    local pattern="$2"
    rg -q --fixed-strings "${pattern}" "${path}" || fail "Expected '${pattern}' in ${path}"
}

assert_equals() {
    local actual="$1"
    local expected="$2"
    local label="$3"
    [[ "${actual}" == "${expected}" ]] || fail "${label}: expected '${expected}', got '${actual}'"
}

assert_number_gt_zero() {
    local value="$1"
    local label="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] || fail "${label}: expected integer, got '${value}'"
    (( value > 0 )) || fail "${label}: expected > 0, got '${value}'"
}

BACKUP_DIR="$(mktemp -d)"
CHAMPION_BACKUP="${BACKUP_DIR}/current.env.bak"
cleanup() {
    if [[ -f "${CHAMPION_BACKUP}" ]]; then
        mkdir -p "${BLACKLIGHT_CHAMPION_DIR}"
        cp "${CHAMPION_BACKUP}" "${BLACKLIGHT_CHAMPION_ENV_PATH}"
    else
        rm -f "${BLACKLIGHT_CHAMPION_ENV_PATH}"
    fi
    rm -rf "${BACKUP_DIR}"
}
trap cleanup EXIT

if [[ -f "${BLACKLIGHT_CHAMPION_ENV_PATH}" ]]; then
    cp "${BLACKLIGHT_CHAMPION_ENV_PATH}" "${CHAMPION_BACKUP}"
fi
rm -f "${BLACKLIGHT_CHAMPION_ENV_PATH}"

BIN_PATH="${ARMAGETRON_SELFPLAY_BIN:-${REPO_ROOT}/src/armagetronad_main}"
[[ -x "${BIN_PATH}" ]] || fail "Missing benchmark binary: ${BIN_PATH}"

TEST_BASE_CFG_REL="examples/trained_ai_champion_test.cfg"

echo "Running helper regression checks..."

TMP_HELPERS_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_HELPERS_DIR}"; cleanup' EXIT

REPORT_A="${TMP_HELPERS_DIR}/report_a.txt"
REPORT_B="${TMP_HELPERS_DIR}/report_b.txt"
cat > "${REPORT_A}" <<'EOF'
candidate_mean_win_rate 0.100000
candidate_mean_distance 100.000000
EOF
cat > "${REPORT_B}" <<'EOF'
candidate_mean_win_rate 0.100000
candidate_mean_distance 120.000000
EOF

BEST_REPORT="$(blacklight_pick_best_report "${REPORT_A}" "${REPORT_B}")"
assert_equals "${BEST_REPORT}" "${REPORT_B}" "best benchmark report"

blacklight_promote_threshold_passes 0.130000 0.100000 100.0 100.0 || fail "expected promotion threshold pass"
! blacklight_promote_threshold_passes 0.110000 0.100000 94.0 100.0 || fail "expected promotion threshold failure"
blacklight_mixed_veto_passes 0.110000 0.120000 92.0 100.0 || fail "expected mixed veto pass"
! blacklight_mixed_veto_passes 0.050000 0.120000 89.0 100.0 || fail "expected mixed veto failure"

echo "Running classic_bootstrap profile smoke..."

BOOTSTRAP_RUN="champion_bootstrap_test_$(date +%Y%m%d-%H%M%S)"
ARMAGETRON_SELFPLAY_BIN="${BIN_PATH}" \
ARMAGETRON_SELFPLAY_BASE_CFG_REL="${TEST_BASE_CFG_REL}" \
ARMAGETRON_SELFPLAY_LIMIT_ROUNDS=1 \
ARMAGETRON_SELFPLAY_SAVE_EVERY=1 \
ARMAGETRON_SELFPLAY_CHECKPOINT_EVERY=1 \
"${SCRIPT_DIR}/blacklight.sh" train \
    --profile classic_bootstrap \
    --name "${BOOTSTRAP_RUN}" \
    --checkpoint-every 1 \
    --parallel-workers 2 \
    --sync-seconds 40 \
    --duration 90

BOOTSTRAP_MANIFEST="${REPO_ROOT}/var/blacklight_runs/${BOOTSTRAP_RUN}/run_manifest.env"
assert_file "${BOOTSTRAP_MANIFEST}"
# shellcheck disable=SC1090
source "${BOOTSTRAP_MANIFEST}"

assert_equals "${PROFILE}" "classic_bootstrap" "bootstrap profile"
assert_equals "${TRAINING_MODE}" "parallel" "bootstrap mode"
assert_nonempty_file "${MODEL_ABS}"
assert_nonempty_file "${METRICS_SUMMARY_ABS}"
assert_nonempty_file "${STATE_FILE_ABS}"

BOOTSTRAP_UPDATES="$(blacklight_model_updates "${MODEL_ABS}")"
assert_number_gt_zero "${BOOTSTRAP_UPDATES}" "bootstrap policy updates"
assert_contains "${STATE_FILE_ABS}" "source_episodes "

BOOTSTRAP_WORKER_CFG="${REPO_ROOT}/config/generated_blacklight_parallel_${BOOTSTRAP_RUN}_worker01.cfg"
assert_file "${BOOTSTRAP_WORKER_CFG}"
assert_contains "${BOOTSTRAP_WORKER_CFG}" "AI_TRAINED_BOT_COUNT 1"
assert_contains "${BOOTSTRAP_WORKER_CFG}" "AI_TRAINED_POLICY_POOL_SIZE 0"
assert_contains "${BOOTSTRAP_WORKER_CFG}" "AI_TRAINED_POLICY_HISTORIC_PROB 0"

BOOTSTRAP_CANDIDATE_COUNT=0
while IFS= read -r bootstrap_candidate; do
    [[ -n "${bootstrap_candidate}" ]] || continue
    BOOTSTRAP_CANDIDATE_COUNT=$(( BOOTSTRAP_CANDIDATE_COUNT + 1 ))
done < <(blacklight_checkpoint_candidates_for_input "${BOOTSTRAP_MANIFEST}" 3)
(( BOOTSTRAP_CANDIDATE_COUNT >= 2 )) || fail "Expected final model plus at least one checkpoint candidate"

echo "Running benchmark smoke and champion default-reference check..."

BENCH_OUTPUT="$(ARMAGETRON_SELFPLAY_BIN="${BIN_PATH}" ARMAGETRON_BENCH_BASE_CFG_REL="${TEST_BASE_CFG_REL}" ARMAGETRON_BENCH_SESSIONS=1 ARMAGETRON_BENCH_DURATION_SECONDS=20 ARMAGETRON_BENCH_LIMIT_ROUNDS=4 "${SCRIPT_DIR}/blacklight.sh" bench --suite classic_primary --candidate "${BOOTSTRAP_MANIFEST}")"
BENCH_REPORT="$(printf '%s\n' "${BENCH_OUTPUT}" | awk '$1 == "report_path" { print $2; exit }')"
assert_file "${BENCH_REPORT}"
assert_contains "${BENCH_REPORT}" "candidate_mean_win_rate "
assert_contains "${BENCH_REPORT}" "candidate_mean_distance "

mkdir -p "${BLACKLIGHT_CHAMPION_DIR}"
cp "${MODEL_ABS}" "${BACKUP_DIR}/champion_probe.txt"
cat > "${BLACKLIGHT_CHAMPION_ENV_PATH}" <<EOF
CHAMPION_MODEL_ABS=${BACKUP_DIR}/champion_probe.txt
SOURCE_MODEL_ABS=${MODEL_ABS}
CLASSIC_BENCH_REPORT_ABS=${BENCH_REPORT}
MIXED_BENCH_REPORT_ABS=${BENCH_REPORT}
PROMOTED_AT=test
EOF

BENCH_WITH_REFERENCE_OUTPUT="$(ARMAGETRON_SELFPLAY_BIN="${BIN_PATH}" ARMAGETRON_BENCH_BASE_CFG_REL="${TEST_BASE_CFG_REL}" ARMAGETRON_BENCH_SESSIONS=1 ARMAGETRON_BENCH_DURATION_SECONDS=20 ARMAGETRON_BENCH_LIMIT_ROUNDS=4 "${SCRIPT_DIR}/blacklight.sh" bench --suite classic_primary --candidate "${BOOTSTRAP_MANIFEST}")"
BENCH_WITH_REFERENCE_REPORT="$(printf '%s\n' "${BENCH_WITH_REFERENCE_OUTPUT}" | awk '$1 == "report_path" { print $2; exit }')"
assert_file "${BENCH_WITH_REFERENCE_REPORT}"
assert_contains "${BENCH_WITH_REFERENCE_REPORT}" "reference_model ${BACKUP_DIR}/champion_probe.txt"

echo "Running classic_hardening resume smoke..."

BOOTSTRAP_PARENT_MODEL="$(blacklight_resolve_model_input "${BOOTSTRAP_MANIFEST}" 1)"
HARDENING_RUN="champion_hardening_test_$(date +%Y%m%d-%H%M%S)"
ARMAGETRON_SELFPLAY_BIN="${BIN_PATH}" \
ARMAGETRON_SELFPLAY_BASE_CFG_REL="${TEST_BASE_CFG_REL}" \
ARMAGETRON_SELFPLAY_LIMIT_ROUNDS=6 \
ARMAGETRON_SELFPLAY_SAVE_EVERY=1 \
ARMAGETRON_SELFPLAY_CHECKPOINT_EVERY=1 \
"${SCRIPT_DIR}/blacklight.sh" train \
    --profile classic_hardening \
    --name "${HARDENING_RUN}" \
    --parallel-workers 1 \
    --sync-seconds 10 \
    --duration 30 \
    --resume "${BOOTSTRAP_MANIFEST}"

HARDENING_MANIFEST="${REPO_ROOT}/var/blacklight_runs/${HARDENING_RUN}/run_manifest.env"
assert_file "${HARDENING_MANIFEST}"
# shellcheck disable=SC1090
source "${HARDENING_MANIFEST}"

assert_equals "${PROFILE}" "classic_hardening" "hardening profile"
assert_equals "${PARENT_MODEL}" "${BOOTSTRAP_PARENT_MODEL}" "hardening parent model"
HARDENING_WORKER_CFG="${REPO_ROOT}/config/generated_blacklight_parallel_${HARDENING_RUN}_worker01.cfg"
assert_contains "${HARDENING_WORKER_CFG}" "AI_TRAINED_BOT_COUNT 2"
assert_contains "${HARDENING_WORKER_CFG}" "AI_TRAINED_POLICY_POOL_SIZE 8"
assert_contains "${HARDENING_WORKER_CFG}" "AI_TRAINED_POLICY_SNAPSHOT_EVERY 40"
assert_contains "${HARDENING_WORKER_CFG}" "AI_TRAINED_POLICY_SNAPSHOT_WARMUP 80"
assert_contains "${HARDENING_WORKER_CFG}" "AI_TRAINED_POLICY_HISTORIC_PROB 0.25"

echo "Running mixed_league resume smoke..."

HARDENING_PARENT_MODEL="$(blacklight_resolve_model_input "${HARDENING_MANIFEST}" 1)"
MIXED_RUN="champion_mixed_test_$(date +%Y%m%d-%H%M%S)"
ARMAGETRON_SELFPLAY_BIN="${BIN_PATH}" \
ARMAGETRON_SELFPLAY_BASE_CFG_REL="${TEST_BASE_CFG_REL}" \
ARMAGETRON_SELFPLAY_LIMIT_ROUNDS=6 \
ARMAGETRON_SELFPLAY_SAVE_EVERY=1 \
ARMAGETRON_SELFPLAY_CHECKPOINT_EVERY=1 \
"${SCRIPT_DIR}/blacklight.sh" train \
    --profile mixed_league \
    --name "${MIXED_RUN}" \
    --parallel-workers 1 \
    --sync-seconds 10 \
    --duration 30 \
    --resume "${HARDENING_MANIFEST}"

MIXED_MANIFEST="${REPO_ROOT}/var/blacklight_runs/${MIXED_RUN}/run_manifest.env"
assert_file "${MIXED_MANIFEST}"
# shellcheck disable=SC1090
source "${MIXED_MANIFEST}"

assert_equals "${PROFILE}" "mixed_league" "mixed profile"
assert_equals "${PARENT_MODEL}" "${HARDENING_PARENT_MODEL}" "mixed parent model"
MIXED_WORKER_CFG="${REPO_ROOT}/config/generated_blacklight_parallel_${MIXED_RUN}_worker01.cfg"
assert_contains "${MIXED_WORKER_CFG}" "AI_TRAINED_BOT_COUNT 4"
assert_contains "${MIXED_WORKER_CFG}" "AI_TRAINED_POLICY_POOL_SIZE 8"
assert_contains "${MIXED_WORKER_CFG}" "AI_TRAINED_POLICY_HISTORIC_PROB 0.35"

echo "Champion pipeline tests passed."
