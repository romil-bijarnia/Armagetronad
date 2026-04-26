#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/blacklight_lib.sh"
REPO_ROOT="${BLACKLIGHT_REPO_ROOT}"

usage() {
    cat <<'EOF'
Blacklight Vast.ai helper

Usage:
  ./scripts/blacklight_vast.sh setup [options]
  ./scripts/blacklight_vast.sh pack [options]
  ./scripts/blacklight_vast.sh apply MODEL_PATH [options]

Setup/pack options:
  --data SOURCE_LIST_OR_RUN
  --output-dir DIR
  --epochs COUNT
  --batch-size COUNT
  --lr RATE
  --max-examples COUNT
  --device auto|cuda|cpu
  --run-name NAME
  --ssh-host USER@HOST
  --ssh-port PORT
  --remote-dir DIR
  --local-output DIR

Apply options:
  --distill
  --alpha RATE
  --replace-current
  --keep-source

The setup/pack command creates a self-contained tar.gz job for a Vast.ai website
SSH/Jupyter instance plus local helper scripts for upload, remote start, watch,
download, apply, and benchmark. The job trains one model forward from the
current local model and writes output/current_model.txt on the Vast machine.
By default, apply installs that trained continuation back into the single local
current model path. Use --distill only for older/external model sources.
EOF
}

absolute_path() {
    local input_path="$1"
    if [[ "${input_path}" == /* ]]; then
        printf '%s\n' "${input_path}"
    else
        printf '%s/%s\n' "$(pwd)" "${input_path}"
    fi
}

copy_teacher_logs() {
    local source_list_abs="$1"
    local job_dir="$2"
    local output_list="${job_dir}/var/blacklight_vast/source_list.txt"
    local index=0
    local record_rel=""
    local record_abs=""
    local dest_rel=""
    local dest_abs=""

    : > "${output_list}"
    while IFS= read -r record_rel; do
        [[ -n "${record_rel}" ]] || continue
        [[ "${record_rel}" != \#* ]] || continue

        if [[ "${record_rel}" == /* ]]; then
            record_abs="${record_rel}"
        else
            record_abs="${REPO_ROOT}/var/${record_rel}"
        fi
        if [[ ! -s "${record_abs}" ]]; then
            echo "Skipping missing/empty teacher log: ${record_abs}" >&2
            continue
        fi

        index=$(( index + 1 ))
        dest_rel="$(printf 'blacklight_vast/teacher_logs/log_%04d.log' "${index}")"
        dest_abs="${job_dir}/var/${dest_rel}"
        mkdir -p "$(dirname "${dest_abs}")"
        cp "${record_abs}" "${dest_abs}"
        printf '%s\n' "${dest_rel}" >> "${output_list}"
    done < "${source_list_abs}"

    if (( index == 0 )); then
        echo "No teacher logs were copied. Run ./scripts/blacklight.sh collect first." >&2
        return 1
    fi

    printf '%s\n' "${index}"
}

write_remote_runner() {
    local job_dir="$1"
    local run_name="$2"
    local device="$3"
    local epochs="$4"
    local batch_size="$5"
    local lr="$6"
    local max_examples="$7"

    cat > "${job_dir}/remote_train.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "\$(dirname "\$0")"

if [[ -n "\${BLACKLIGHT_PYTHON:-}" ]]; then
    PYTHON_BIN="\${BLACKLIGHT_PYTHON}"
elif [[ -x /venv/main/bin/python ]] && /venv/main/bin/python - <<'PY' >/dev/null 2>&1
import torch
PY
then
    PYTHON_BIN="/venv/main/bin/python"
else
    PYTHON_BIN="python3"
fi
DEVICE="\${BLACKLIGHT_VAST_DEVICE:-${device}}"
EPOCHS="\${BLACKLIGHT_VAST_EPOCHS:-${epochs}}"
BATCH_SIZE="\${BLACKLIGHT_VAST_BATCH_SIZE:-${batch_size}}"
LR="\${BLACKLIGHT_VAST_LR:-${lr}}"
MAX_EXAMPLES="\${BLACKLIGHT_VAST_MAX_EXAMPLES:-${max_examples}}"
RUN_NAME="\${BLACKLIGHT_VAST_RUN_NAME:-${run_name}}"

if ! "\${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import torch
PY
then
    if ! "\${PYTHON_BIN}" -m pip install --upgrade pip; then
        echo "PyTorch is not importable from \${PYTHON_BIN}, and pip upgrade failed." >&2
        echo "Set BLACKLIGHT_PYTHON=/venv/main/bin/python or use a Vast PyTorch template." >&2
        exit 1
    fi
    if ! "\${PYTHON_BIN}" -m pip install torch numpy; then
        echo "PyTorch install failed for \${PYTHON_BIN}." >&2
        echo "Set BLACKLIGHT_PYTHON=/venv/main/bin/python or use a Vast PyTorch template." >&2
        exit 1
    fi
fi

args=(
    scripts/blacklight_gpu_teacher_train.py
    --repo-root "\$PWD"
    --source-list "\$PWD/var/blacklight_vast/source_list.txt"
    --initial-model "\$PWD/var/blacklight_champions/current_model.txt"
    --parent-model "\$PWD/var/blacklight_champions/current_model.txt"
    --generation blacklight_vast_train
    --run-name "\${RUN_NAME}"
    --device "\${DEVICE}"
    --epochs "\${EPOCHS}"
    --batch-size "\${BATCH_SIZE}"
    --lr "\${LR}"
    --checkpoint-every 0
)
if [[ -n "\${MAX_EXAMPLES}" ]]; then
    args+=( --max-examples "\${MAX_EXAMPLES}" )
fi

"\${PYTHON_BIN}" "\${args[@]}"

mkdir -p output
cp "var/blacklight_runs/\${RUN_NAME}/trained_ai_teacher_cnn_model.txt" output/current_model.txt
cp "var/blacklight_runs/\${RUN_NAME}/trained_ai_training_metrics.csv" output/training_metrics.csv
cp "var/blacklight_runs/\${RUN_NAME}/trained_ai_training_metrics.csv.latest" output/training_metrics.latest
cp "var/blacklight_runs/\${RUN_NAME}/training_events.log" output/training_events.log

if command -v sha256sum >/dev/null 2>&1; then
    sha256sum output/current_model.txt > output/current_model.sha256
fi

echo "Blacklight Vast.ai job complete"
echo "model output/current_model.txt"
echo "metrics output/training_metrics.latest"
EOF
    chmod +x "${job_dir}/remote_train.sh"
}

write_job_readme() {
    local job_dir="$1"
    local archive_name="$2"

    cat > "${job_dir}/README_VAST.md" <<EOF
# Blacklight Vast.ai Training Job

This bundle trains the single Blacklight current model forward. It does not
select from many checkpoints.

On the Vast.ai machine:

\`\`\`bash
tar -xzf ${archive_name}
cd $(basename "${job_dir}")
./remote_train.sh
\`\`\`

After it finishes, copy this file back to the local Armagetron repo:

\`\`\`text
output/current_model.txt
\`\`\`

Then apply it locally as the trained continuation of the current champion:

\`\`\`bash
./scripts/blacklight.sh vast apply /path/to/current_model.txt
\`\`\`
EOF
}

write_control_kit() {
    local job_dir="$1"
    local archive_path="$2"
    local ssh_host="$3"
    local ssh_port="$4"
    local remote_dir="$5"
    local local_output="$6"
    local helper_dir="${job_dir}/local_vast_control"
    local job_name=""
    local archive_name=""

    job_name="$(basename "${job_dir}")"
    archive_name="$(basename "${archive_path}")"
    mkdir -p "${helper_dir}"

    {
        blacklight_env_line "BLACKLIGHT_VAST_HOST" "${ssh_host:-root@VAST_HOST}"
        blacklight_env_line "BLACKLIGHT_VAST_PORT" "${ssh_port:-VAST_PORT}"
        blacklight_env_line "BLACKLIGHT_VAST_REMOTE_DIR" "${remote_dir}"
        blacklight_env_line "BLACKLIGHT_VAST_ARCHIVE" "${archive_path}"
        blacklight_env_line "BLACKLIGHT_VAST_ARCHIVE_NAME" "${archive_name}"
        blacklight_env_line "BLACKLIGHT_VAST_JOB_NAME" "${job_name}"
        blacklight_env_line "BLACKLIGHT_VAST_LOCAL_OUTPUT" "${local_output}"
        blacklight_env_line "BLACKLIGHT_REPO_ROOT" "${REPO_ROOT}"
    } > "${helper_dir}/vast.env"

    cat > "${helper_dir}/lib.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

CONTROL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${CONTROL_DIR}/vast.env"

require_vast_connection() {
    if [[ -z "${BLACKLIGHT_VAST_HOST:-}" || "${BLACKLIGHT_VAST_HOST}" == "root@VAST_HOST" ]]; then
        echo "Edit ${CONTROL_DIR}/vast.env and set BLACKLIGHT_VAST_HOST from the Vast SSH command." >&2
        exit 1
    fi
    if [[ -z "${BLACKLIGHT_VAST_PORT:-}" || "${BLACKLIGHT_VAST_PORT}" == "VAST_PORT" ]]; then
        echo "Edit ${CONTROL_DIR}/vast.env and set BLACKLIGHT_VAST_PORT from the Vast SSH command." >&2
        exit 1
    fi
}

remote_job_dir() {
    printf '%s/%s\n' "${BLACKLIGHT_VAST_REMOTE_DIR%/}" "${BLACKLIGHT_VAST_JOB_NAME}"
}
EOF
    chmod +x "${helper_dir}/lib.sh"

    cat > "${helper_dir}/01_upload.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
require_vast_connection

ssh -p "${BLACKLIGHT_VAST_PORT}" "${BLACKLIGHT_VAST_HOST}" "mkdir -p '${BLACKLIGHT_VAST_REMOTE_DIR}'"
scp -P "${BLACKLIGHT_VAST_PORT}" "${BLACKLIGHT_VAST_ARCHIVE}" "${BLACKLIGHT_VAST_HOST}:${BLACKLIGHT_VAST_REMOTE_DIR}/"

echo "uploaded ${BLACKLIGHT_VAST_ARCHIVE_NAME} to ${BLACKLIGHT_VAST_HOST}:${BLACKLIGHT_VAST_REMOTE_DIR}"
EOF

    cat > "${helper_dir}/02_start_training.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
require_vast_connection

ssh -p "${BLACKLIGHT_VAST_PORT}" "${BLACKLIGHT_VAST_HOST}" "
set -e
cd '${BLACKLIGHT_VAST_REMOTE_DIR}'
tar -xzf '${BLACKLIGHT_VAST_ARCHIVE_NAME}'
cd '${BLACKLIGHT_VAST_JOB_NAME}'
nohup ./remote_train.sh > remote_train.out 2>&1 &
echo \$! > remote_train.pid
echo started \$(cat remote_train.pid)
echo log \$(pwd)/remote_train.out
"
EOF

    cat > "${helper_dir}/03_watch.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
require_vast_connection

ssh -p "${BLACKLIGHT_VAST_PORT}" "${BLACKLIGHT_VAST_HOST}" "cd '$(remote_job_dir)' && tail -n 80 -f remote_train.out"
EOF

    cat > "${helper_dir}/04_download.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
require_vast_connection

mkdir -p "${BLACKLIGHT_VAST_LOCAL_OUTPUT}"
scp -P "${BLACKLIGHT_VAST_PORT}" -r "${BLACKLIGHT_VAST_HOST}:$(remote_job_dir)/output" "${BLACKLIGHT_VAST_LOCAL_OUTPUT}/"

echo "downloaded output to ${BLACKLIGHT_VAST_LOCAL_OUTPUT}/output"
echo "model ${BLACKLIGHT_VAST_LOCAL_OUTPUT}/output/current_model.txt"
EOF

    cat > "${helper_dir}/05_apply.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

model_path="${BLACKLIGHT_VAST_LOCAL_OUTPUT}/output/current_model.txt"
if [[ ! -f "${model_path}" ]]; then
    echo "Model is not downloaded yet: ${model_path}" >&2
    echo "Run 04_download.sh first." >&2
    exit 1
fi

cd "${BLACKLIGHT_REPO_ROOT}"
./scripts/blacklight.sh vast apply "${model_path}"
EOF

    cat > "${helper_dir}/06_benchmark.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

cd "${BLACKLIGHT_REPO_ROOT}"
./scripts/blacklight.sh bench --suite classic_primary --candidate current --sessions 3 --duration 45 --rounds 30
EOF

    chmod +x "${helper_dir}"/0*.sh

    cat > "${helper_dir}/README_LOCAL.md" <<EOF
# Blacklight Vast.ai Local Control Kit

This folder controls one Vast.ai training job for one Blacklight current model.
The apply step installs the downloaded output into the current local champion
path after making a rollback backup.

1. Rent one Vast SSH/Jupyter instance from the website.
2. Edit \`vast.env\` and fill in \`BLACKLIGHT_VAST_HOST\` plus
   \`BLACKLIGHT_VAST_PORT\` from the Vast SSH command.
3. Run:

\`\`\`bash
./01_upload.sh
./02_start_training.sh
./03_watch.sh
./04_download.sh
./05_apply.sh
./06_benchmark.sh
\`\`\`

The remote job writes:

\`\`\`text
${remote_dir%/}/${job_name}/output/current_model.txt
\`\`\`

The local downloaded model path is:

\`\`\`text
${local_output}/output/current_model.txt
\`\`\`
EOF

    printf '%s\n' "${helper_dir}"
}

run_pack() {
    local data_source=""
    local output_dir="${REPO_ROOT}/var/blacklight_vast"
    local epochs="4"
    local batch_size="4096"
    local lr="0.002"
    local max_examples=""
    local device="auto"
    local run_name=""
    local ssh_host=""
    local ssh_port=""
    local remote_dir="/workspace"
    local local_output=""
    local source_list_abs=""
    local current_model=""
    local job_id=""
    local job_dir=""
    local archive_path=""
    local copied_logs="0"
    local helper_dir=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --data)
                data_source="$2"
                shift 2
                ;;
            --output-dir)
                output_dir="$(absolute_path "$2")"
                shift 2
                ;;
            --epochs)
                epochs="$2"
                shift 2
                ;;
            --batch-size)
                batch_size="$2"
                shift 2
                ;;
            --lr)
                lr="$2"
                shift 2
                ;;
            --max-examples)
                max_examples="$2"
                shift 2
                ;;
            --device)
                device="$2"
                shift 2
                ;;
            --run-name)
                run_name="$2"
                shift 2
                ;;
            --ssh-host)
                ssh_host="$2"
                shift 2
                ;;
            --ssh-port)
                ssh_port="$2"
                shift 2
                ;;
            --remote-dir)
                remote_dir="$2"
                shift 2
                ;;
            --local-output)
                local_output="$(absolute_path "$2")"
                shift 2
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                echo "Unknown pack option: $1" >&2
                usage >&2
                exit 1
                ;;
        esac
    done

    if ! source_list_abs="$(blacklight_resolve_source_list_input "${data_source}")"; then
        echo "Could not resolve teacher data source. Run collect first or pass --data." >&2
        exit 1
    fi
    if ! current_model="$(blacklight_resolve_model_input current 1)"; then
        echo "Could not resolve current Blacklight model." >&2
        exit 1
    fi

    job_id="blacklight_vast_$(date +%Y%m%d-%H%M%S)"
    [[ -n "${run_name}" ]] || run_name="${job_id}_train"
    job_dir="${output_dir}/${job_id}"
    archive_path="${output_dir}/${job_id}.tar.gz"
    [[ -n "${local_output}" ]] || local_output="${output_dir}/${job_id}_result"

    rm -rf "${job_dir}"
    mkdir -p "${job_dir}/scripts" "${job_dir}/var/blacklight_champions" "${job_dir}/var/blacklight_vast"
    cp "${SCRIPT_DIR}/blacklight_gpu_teacher_train.py" "${job_dir}/scripts/blacklight_gpu_teacher_train.py"
    cp "${current_model}" "${job_dir}/var/blacklight_champions/current_model.txt"
    copied_logs="$(copy_teacher_logs "${source_list_abs}" "${job_dir}")"

    write_remote_runner "${job_dir}" "${run_name}" "${device}" "${epochs}" "${batch_size}" "${lr}" "${max_examples}"
    write_job_readme "${job_dir}" "$(basename "${archive_path}")"

    mkdir -p "${output_dir}"
    tar -C "${output_dir}" -czf "${archive_path}" "$(basename "${job_dir}")"
    helper_dir="$(write_control_kit "${job_dir}" "${archive_path}" "${ssh_host}" "${ssh_port}" "${remote_dir}" "${local_output}")"

    echo "Blacklight Vast.ai bundle"
    echo "archive ${archive_path}"
    echo "job_dir ${job_dir}"
    echo "control_kit ${helper_dir}"
    echo "teacher_logs ${copied_logs}"
    echo "current_model ${current_model}"
    echo "epochs ${epochs}"
    echo "batch_size ${batch_size}"
    echo "remote_run cd /workspace && tar -xzf $(basename "${archive_path}") && cd $(basename "${job_dir}") && ./remote_train.sh"
    echo "next_step edit ${helper_dir}/vast.env then run ${helper_dir}/01_upload.sh"
    echo "apply_after_download ./scripts/blacklight.sh vast apply /path/to/current_model.txt"
}

run_apply() {
    local model_path=""
    local installed=""
    local alpha="0.10"
    local distill_current="0"
    local keep_source="0"
    local model_abs=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --alpha)
                alpha="$2"
                shift 2
                ;;
            --distill)
                distill_current="1"
                shift
                ;;
            --replace-current)
                distill_current="0"
                shift
                ;;
            --keep-source)
                keep_source="1"
                shift
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                if [[ -z "${model_path}" ]]; then
                    model_path="$1"
                    shift
                else
                    echo "Unknown apply option: $1" >&2
                    usage >&2
                    exit 1
                fi
                ;;
        esac
    done

    if [[ -z "${model_path}" ]]; then
        echo "apply needs a model path." >&2
        usage >&2
        exit 1
    fi
    if [[ ! -f "${model_path}" ]]; then
        echo "Model file not found: ${model_path}" >&2
        exit 1
    fi

    model_abs="$(absolute_path "${model_path}")"
    if [[ "${distill_current}" == "1" ]]; then
        "${REPO_ROOT}/scripts/blacklight.sh" current distill --from "${model_abs}" --alpha "${alpha}"
        return 0
    fi

    installed="$(blacklight_install_current_model "${model_abs}" "vast-ai" "vast-current-train")"
    echo "Blacklight current champion updated from Vast.ai training"
    echo "source_model ${model_abs}"
    echo "current_model ${installed}"
    if blacklight_source_champion_registry >/dev/null 2>&1; then
        echo "backup_model ${CURRENT_BACKUP_MODEL_ABS:-}"
    fi
    if [[ "${keep_source}" != "1" && "${model_abs}" == "${REPO_ROOT}/var/blacklight_vast/"* ]]; then
        rm -f "${model_abs}"
        echo "source_model_deleted ${model_abs}"
    fi
}

cmd="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${cmd}" in
    setup)
        run_pack "$@"
        ;;
    pack)
        run_pack "$@"
        ;;
    apply)
        run_apply "$@"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown Vast.ai command: ${cmd}" >&2
        usage >&2
        exit 1
        ;;
esac
