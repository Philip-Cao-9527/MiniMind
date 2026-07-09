#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"

python_bin="${project_root}/.venv/bin/python"
train_dir="${project_root}/trainer"
logs_dir="${project_root}/experiments/logs"
out_dir="${project_root}/out"
checkpoints_dir="${project_root}/checkpoints"
data_path="${project_root}/dataset/sft_t2t_mini.jsonl"
pretrain_weight="${out_dir}/pretrain_768.pth"
writer_lock="${checkpoints_dir}/full_sft_subset_768.writer.lock"
manifest_path="${logs_dir}/full-sft-subset-current-run.env"
monitor_script="${project_root}/scripts/monitor_full_sft_subset_memory.sh"

subset_epochs="${SFT_SUBSET_EPOCHS:-2}"
subset_max_samples="${SFT_SUBSET_MAX_SAMPLES:-100000}"
subset_ratio="${SFT_SUBSET_RATIO:-1.0}"
subset_seed="${SFT_SUBSET_SEED:-42}"
subset_mode="${SFT_SUBSET_MODE:-random}"
subset_use_swanlab="${SFT_SUBSET_USE_SWANLAB:-1}"
subset_wandb_project="${SFT_SUBSET_WANDB_PROJECT:-MiniMind-Full-SFT-Subset}"

if [[ "${subset_use_swanlab}" != "0" && "${subset_use_swanlab}" != "1" ]]; then
  printf 'ERROR: SFT_SUBSET_USE_SWANLAB must be 0 or 1, got: %s\n' "${subset_use_swanlab}" >&2
  exit 1
fi

run_id="$(date +%Y%m%d-%H%M%S)"
run_epoch_label="e${subset_epochs}"
session_name="minimind-full-sft-subset-dense768-${run_epoch_label}-${run_id}"
log_path="${logs_dir}/full-sft-subset-dense-768-${run_epoch_label}-${run_id}.log"
monitor_csv_path="${logs_dir}/full-sft-subset-dense-768-${run_epoch_label}-${run_id}-memory.csv"
runner_script="$(mktemp /tmp/minimind-full-sft-subset-runner-XXXXXX.sh)"

cleanup_runner_on_error() {
  rm -f "${runner_script}"
}
trap cleanup_runner_on_error EXIT

require_command() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    printf 'ERROR: missing required command: %s\n' "${name}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    printf 'ERROR: missing %s: %s\n' "${label}" "${path}" >&2
    exit 1
  fi
}

ensure_absent() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    printf 'ERROR: refusing to start because path already exists: %s\n' "${path}" >&2
    exit 1
  fi
}

find_train_pids() {
  ps -eo pid=,comm=,args= | awk '
    $2 ~ /^python([0-9.]*)?$/ && $0 ~ /train_full_sft[.]py/ {
      print $1
    }
  '
}

mkdir -p "${logs_dir}" "${out_dir}" "${checkpoints_dir}"

require_command screen
require_command flock
require_file "${python_bin}" "python interpreter"
require_file "${pretrain_weight}" "pretrain weight"
require_file "${data_path}" "SFT dataset"

ensure_absent "${out_dir}/full_sft_subset_768.pth"
ensure_absent "${checkpoints_dir}/full_sft_subset_768.pth"
ensure_absent "${checkpoints_dir}/full_sft_subset_768_resume.pth"
ensure_absent "${checkpoints_dir}/full_sft_subset_768.pth.tmp"
ensure_absent "${checkpoints_dir}/full_sft_subset_768_resume.pth.tmp"
ensure_absent "${log_path}"
ensure_absent "${monitor_csv_path}"

mapfile -t train_pids < <(find_train_pids)
if [[ "${#train_pids[@]}" -ne 0 ]]; then
  printf 'ERROR: detected existing train_full_sft.py writer(s): %s\n' "${train_pids[*]}" >&2
  exit 1
fi

if screen -ls 2>/dev/null | grep -F "${session_name}" >/dev/null; then
  printf 'ERROR: screen session already exists: %s\n' "${session_name}" >&2
  exit 1
fi

exec {preflight_lock_fd}> "${writer_lock}"
if ! flock -n "${preflight_lock_fd}"; then
  printf 'ERROR: writer lock is already held: %s\n' "${writer_lock}" >&2
  exit 1
fi
flock -u "${preflight_lock_fd}"
exec {preflight_lock_fd}>&-

cat > "${runner_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
trap 'rm -f "${runner_script}"' EXIT

project_root="${project_root}"
train_dir="${train_dir}"
python_bin="${python_bin}"
log_path="${log_path}"
writer_lock="${writer_lock}"

cd "\${train_dir}"

exec {lock_fd}> "\${writer_lock}"
if ! flock -n "\${lock_fd}"; then
  printf '[ERROR] writer lock is already held: %s\n' "\${writer_lock}" | tee -a "\${log_path}" >&2
  exit 23
fi

printf '[INFO] run_id=%s\n' "${run_id}" | tee -a "\${log_path}"
printf '[INFO] writer_lock=%s\n' "\${writer_lock}" | tee -a "\${log_path}"
printf '[INFO] log_path=%s\n' "\${log_path}" | tee -a "\${log_path}"
printf '[INFO] save_weight=full_sft_subset\n' | tee -a "\${log_path}"
printf '[INFO] from_weight=pretrain\n' | tee -a "\${log_path}"
printf '[INFO] from_resume=0\n' | tee -a "\${log_path}"
printf '[INFO] epochs=%s\n' "${subset_epochs}" | tee -a "\${log_path}"
printf '[INFO] use_swanlab=%s\n' "${subset_use_swanlab}" | tee -a "\${log_path}"
printf '[INFO] wandb_project=%s\n' "${subset_wandb_project}" | tee -a "\${log_path}"
printf '[INFO] max_train_samples=%s\n' "${subset_max_samples}" | tee -a "\${log_path}"
printf '[INFO] train_sample_ratio=%s\n' "${subset_ratio}" | tee -a "\${log_path}"
printf '[INFO] train_subset_seed=%s\n' "${subset_seed}" | tee -a "\${log_path}"
printf '[INFO] train_subset_mode=%s\n' "${subset_mode}" | tee -a "\${log_path}"

wandb_args=()
if [[ "${subset_use_swanlab}" == "1" ]]; then
  wandb_args+=(--use_wandb --wandb_project "${subset_wandb_project}")
fi

set -o pipefail
"\${python_bin}" -u train_full_sft.py \\
  --save_dir ../out \\
  --save_weight full_sft_subset \\
  --epochs ${subset_epochs} \\
  --batch_size 1 \\
  --max_seq_len 384 \\
  --accumulation_steps 6 \\
  --num_workers 0 \\
  --learning_rate 1e-5 \\
  --grad_clip 1.0 \\
  --log_interval 20 \\
  --save_interval 10000 \\
  --dtype bfloat16 \\
  --hidden_size 768 \\
  --num_hidden_layers 8 \\
  --use_moe 0 \\
  --from_weight pretrain \\
  --from_resume 0 \\
  --max_train_samples ${subset_max_samples} \\
  --train_sample_ratio ${subset_ratio} \\
  --train_subset_seed ${subset_seed} \\
  --train_subset_mode ${subset_mode} \\
  "\${wandb_args[@]}" \\
  2>&1 | tee -a "\${log_path}"
EOF
chmod +x "${runner_script}"

screen -dmS "${session_name}" bash "${runner_script}"
sleep 1

if ! screen -ls 2>/dev/null | grep -F "${session_name}" >/dev/null; then
  printf 'ERROR: screen session did not stay alive: %s\n' "${session_name}" >&2
  if [[ -f "${log_path}" ]]; then
    printf 'Startup log:\n' >&2
    sed -n '1,120p' "${log_path}" >&2
  fi
  exit 1
fi

cat > "${manifest_path}" <<EOF
RUN_ID='${run_id}'
SCREEN_SESSION='${session_name}'
LOG_PATH='${log_path}'
MONITOR_CSV_PATH='${monitor_csv_path}'
WRITER_LOCK='${writer_lock}'
CHECKPOINT_PATH='${checkpoints_dir}/full_sft_subset_768_resume.pth'
PROJECT_ROOT='${project_root}'
START_SCRIPT='${project_root}/scripts/start_full_sft_dense768_subset_e1.sh'
MONITOR_SCRIPT='${monitor_script}'
SAVE_WEIGHT='full_sft_subset'
FROM_WEIGHT='pretrain'
FROM_RESUME='0'
MAX_TRAIN_SAMPLES='${subset_max_samples}'
TRAIN_SAMPLE_RATIO='${subset_ratio}'
TRAIN_SUBSET_SEED='${subset_seed}'
TRAIN_SUBSET_MODE='${subset_mode}'
EPOCHS='${subset_epochs}'
USE_SWANLAB='${subset_use_swanlab}'
WANDB_PROJECT='${subset_wandb_project}'
EOF

trap - EXIT

printf 'STARTED_SCREEN=%s\n' "${session_name}"
printf 'LOG_PATH=%s\n' "${log_path}"
printf 'MONITOR_CSV_PATH=%s\n' "${monitor_csv_path}"
printf 'MANIFEST_PATH=%s\n' "${manifest_path}"
printf 'WRITER_LOCK=%s\n' "${writer_lock}"
printf 'MAX_TRAIN_SAMPLES=%s\n' "${subset_max_samples}"
printf 'EPOCHS=%s\n' "${subset_epochs}"
printf 'USE_SWANLAB=%s\n' "${subset_use_swanlab}"
printf 'WANDB_PROJECT=%s\n' "${subset_wandb_project}"
printf 'SAVE_WEIGHT=full_sft_subset\n'
printf '\n'
printf 'Recommended Windows PowerShell follow-ups:\n'
printf 'wsl -d Ubuntu-24.04 -- cat %s\n' "${manifest_path}"
printf 'wsl -d Ubuntu-24.04 -- screen -ls\n'
printf 'wsl -d Ubuntu-24.04 -- bash %s\n' "${monitor_script}"