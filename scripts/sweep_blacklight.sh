#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/blacklight_lib.sh"
REPO_ROOT="${BLACKLIGHT_REPO_ROOT}"

PROFILE_CHAIN="classic"
SWEEP_ID="${ARMAGETRON_SWEEP_ID:-$(date +%Y%m%d-%H%M%S)}"
SWEEP_NAME="${ARMAGETRON_SWEEP_NAME:-blacklight_${PROFILE_CHAIN}_${SWEEP_ID}}"
SWEEP_DIR_ABS="${BLACKLIGHT_SWEEP_ROOT}/${SWEEP_NAME}"
BOOTSTRAP_REPLICATES="${ARMAGETRON_SWEEP_BOOTSTRAP_REPLICATES:-3}"
HARDENING_REPLICATES="${ARMAGETRON_SWEEP_HARDENING_REPLICATES:-3}"
MIXED_REPLICATES="${ARMAGETRON_SWEEP_MIXED_REPLICATES:-2}"

usage() {
    cat <<'EOF'
Usage:
  ./scripts/sweep_blacklight.sh --profile-chain classic
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile-chain)
            PROFILE_CHAIN="$2"
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

if [[ "${PROFILE_CHAIN}" != "classic" ]]; then
    echo "Unsupported profile chain: ${PROFILE_CHAIN}" >&2
    exit 1
fi

mkdir -p "${SWEEP_DIR_ABS}"

declare -a BOOTSTRAP_RUNS=()
declare -a HARDENING_RUNS=()
declare -a MIXED_RUNS=()
declare -a BOOTSTRAP_REPORTS=()
declare -a HARDENING_REPORTS=()
declare -a MIXED_REPORTS=()
declare -A CANDIDATE_MANIFESTS=()

run_training_stage() {
    local profile="$1"
    local stage_name="$2"
    local replicate_index="$3"
    local resume_model="${4:-}"
    local run_name="${SWEEP_NAME}_${stage_name}_r$(printf '%02d' "${replicate_index}")"
    local manifest_path="${REPO_ROOT}/var/blacklight_runs/${run_name}/run_manifest.env"

    if [[ -n "${resume_model}" ]]; then
        "${SCRIPT_DIR}/blacklight.sh" train --profile "${profile}" --name "${run_name}" --resume "${resume_model}"
    else
        "${SCRIPT_DIR}/blacklight.sh" train --profile "${profile}" --name "${run_name}"
    fi

    [[ -f "${manifest_path}" ]] || { echo "Missing manifest after ${profile} run: ${manifest_path}" >&2; exit 1; }
    printf '%s\n' "${manifest_path}"
}

benchmark_candidate_model() {
    local candidate_model="$1"
    local output
    local report_path

    output="$("${SCRIPT_DIR}/benchmark_blacklight.sh" --suite classic_primary --candidate "${candidate_model}")"
    printf '%s\n' "${output}" >&2
    report_path="$(printf '%s\n' "${output}" | awk '$1 == "report_path" { print $2; exit }')"
    [[ -n "${report_path}" && -f "${report_path}" ]] || {
        echo "Benchmark for candidate ${candidate_model} did not produce a report." >&2
        exit 1
    }
    printf '%s\n' "${report_path}"
}

collect_stage_candidates() {
    local manifest_path="$1"
    local report_array_name="$2"
    local candidate_model
    local bench_report

    while IFS= read -r candidate_model; do
        [[ -n "${candidate_model}" ]] || continue
        CANDIDATE_MANIFESTS["${candidate_model}"]="${manifest_path}"
        bench_report="$(benchmark_candidate_model "${candidate_model}")"
        eval "${report_array_name}+=(\"\${bench_report}\")"
    done < <(blacklight_checkpoint_candidates_for_input "${manifest_path}" 3)
}

best_model_from_reports() {
    local best_report
    best_report="$(blacklight_pick_best_report "$@")"
    printf '%s|%s\n' "${best_report}" "$(blacklight_report_value "${best_report}" candidate_model)"
}

for (( i = 1; i <= BOOTSTRAP_REPLICATES; ++i )); do
    BOOTSTRAP_RUNS+=( "$(run_training_stage classic_bootstrap bootstrap "${i}")" )
done

for manifest_path in "${BOOTSTRAP_RUNS[@]}"; do
    collect_stage_candidates "${manifest_path}" BOOTSTRAP_REPORTS
done

IFS='|' read -r BOOTSTRAP_BEST_REPORT BOOTSTRAP_BEST_MODEL < <(best_model_from_reports "${BOOTSTRAP_REPORTS[@]}")
if [[ -n "${CANDIDATE_MANIFESTS[${BOOTSTRAP_BEST_MODEL}]:-}" ]]; then
    blacklight_env_set "${CANDIDATE_MANIFESTS[${BOOTSTRAP_BEST_MODEL}]}" "CHOSEN_CHECKPOINT_ABS" "${BOOTSTRAP_BEST_MODEL}"
    blacklight_env_set "${CANDIDATE_MANIFESTS[${BOOTSTRAP_BEST_MODEL}]}" "BENCH_REPORT_ABS" "${BOOTSTRAP_BEST_REPORT}"
fi

for (( i = 1; i <= HARDENING_REPLICATES; ++i )); do
    HARDENING_RUNS+=( "$(run_training_stage classic_hardening hardening "${i}" "${BOOTSTRAP_BEST_MODEL}")" )
done

for manifest_path in "${HARDENING_RUNS[@]}"; do
    collect_stage_candidates "${manifest_path}" HARDENING_REPORTS
done

IFS='|' read -r HARDENING_BEST_REPORT HARDENING_BEST_MODEL < <(best_model_from_reports "${HARDENING_REPORTS[@]}")
if [[ -n "${CANDIDATE_MANIFESTS[${HARDENING_BEST_MODEL}]:-}" ]]; then
    blacklight_env_set "${CANDIDATE_MANIFESTS[${HARDENING_BEST_MODEL}]}" "CHOSEN_CHECKPOINT_ABS" "${HARDENING_BEST_MODEL}"
    blacklight_env_set "${CANDIDATE_MANIFESTS[${HARDENING_BEST_MODEL}]}" "BENCH_REPORT_ABS" "${HARDENING_BEST_REPORT}"
fi

for (( i = 1; i <= MIXED_REPLICATES; ++i )); do
    MIXED_RUNS+=( "$(run_training_stage mixed_league mixed "${i}" "${HARDENING_BEST_MODEL}")" )
done

for manifest_path in "${MIXED_RUNS[@]}"; do
    collect_stage_candidates "${manifest_path}" MIXED_REPORTS
done

IFS='|' read -r FINAL_BEST_REPORT FINAL_BEST_MODEL < <(best_model_from_reports "${MIXED_REPORTS[@]}")
if [[ -n "${CANDIDATE_MANIFESTS[${FINAL_BEST_MODEL}]:-}" ]]; then
    blacklight_env_set "${CANDIDATE_MANIFESTS[${FINAL_BEST_MODEL}]}" "CHOSEN_CHECKPOINT_ABS" "${FINAL_BEST_MODEL}"
    blacklight_env_set "${CANDIDATE_MANIFESTS[${FINAL_BEST_MODEL}]}" "BENCH_REPORT_ABS" "${FINAL_BEST_REPORT}"
fi

PROMOTION_OUTPUT="$("${SCRIPT_DIR}/promote_blacklight.sh" --candidate "${FINAL_BEST_MODEL}")"
printf '%s\n' "${PROMOTION_OUTPUT}" >&2
PROMOTION_STATUS="$(printf '%s\n' "${PROMOTION_OUTPUT}" | awk '$1 == "promotion_status" { print $2; exit }')"
PROMOTED_CHAMPION="$(printf '%s\n' "${PROMOTION_OUTPUT}" | awk '$1 == "champion_model" { print $2; exit }')"

cat > "${SWEEP_DIR_ABS}/summary.env" <<EOF
SWEEP_NAME=${SWEEP_NAME}
PROFILE_CHAIN=${PROFILE_CHAIN}
BOOTSTRAP_BEST_MODEL_ABS=${BOOTSTRAP_BEST_MODEL}
BOOTSTRAP_BEST_REPORT_ABS=${BOOTSTRAP_BEST_REPORT}
HARDENING_BEST_MODEL_ABS=${HARDENING_BEST_MODEL}
HARDENING_BEST_REPORT_ABS=${HARDENING_BEST_REPORT}
FINAL_BEST_MODEL_ABS=${FINAL_BEST_MODEL}
FINAL_BEST_REPORT_ABS=${FINAL_BEST_REPORT}
PROMOTION_STATUS=${PROMOTION_STATUS}
PROMOTED_CHAMPION_ABS=${PROMOTED_CHAMPION}
EOF

echo "Blacklight sweep"
echo "profile_chain ${PROFILE_CHAIN}"
echo "sweep_dir ${SWEEP_DIR_ABS}"
echo "bootstrap_best_model ${BOOTSTRAP_BEST_MODEL}"
echo "bootstrap_best_report ${BOOTSTRAP_BEST_REPORT}"
echo "hardening_best_model ${HARDENING_BEST_MODEL}"
echo "hardening_best_report ${HARDENING_BEST_REPORT}"
echo "final_best_model ${FINAL_BEST_MODEL}"
echo "final_best_report ${FINAL_BEST_REPORT}"
echo "promotion_status ${PROMOTION_STATUS}"
if [[ -n "${PROMOTED_CHAMPION}" ]]; then
    echo "champion_model ${PROMOTED_CHAMPION}"
fi
