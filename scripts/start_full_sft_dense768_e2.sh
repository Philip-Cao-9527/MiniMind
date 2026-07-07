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
writer_lock="${checkpoints_dir}/full_sft_768.writer.lock"
manifest_path="${logs_dir}/full-sft-current-run.env"
monitor_script="${project_root}/scripts/monitor_full_sft_memory.sh"

run_id="$(date +%Y%m%d-%H%M%S)"
session_name="minimind-full-sft-dense768-e2-${run_id}"
log_path="${logs_dir}/full-sft-dense-768-e2-${run_id}.log"
monitor_csv_path="${logs_dir}/full-sft-dense-768-e2-${run_id}-memory.csv"
runner_script="$(mktemp /tmp/minimind-full-sft-runner-XXXXXX.sh)"

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

ensure_absent "${out_dir}/full_sft_768.pth"
ensure_absent "${checkpoints_dir}/full_sft_768.pth"
ensure_absent "${checkpoints_dir}/full_sft_768_resume.pth"
ensure_absent "${checkpoints_dir}/full_sft_768.pth.tmp"
ensure_absent "${checkpoints_dir}/full_sft_768_resume.pth.tmp"
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

set -o pipefail
"\${python_bin}" -u train_full_sft.py \\
  --save_dir ../out \\
  --save_weight full_sft \\
  --epochs 2 \\
  --batch_size 1 \\
  --max_seq_len 384 \\
  --accumulation_steps 6 \\
  --num_workers 0 \\
  --learning_rate 1e-5 \\
  --grad_clip 1.0 \\
  --log_interval 20 \\
  --save_interval 5000 \\
  --use_wandb \\
  --wandb_project MiniMind-Full-SFT \\
  --dtype bfloat16 \\
  --hidden_size 768 \\
  --num_hidden_layers 8 \\
  --use_moe 0 \\
  --from_weight pretrain \\
  --from_resume 0 \\
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
PROJECT_ROOT='${project_root}'
START_SCRIPT='${project_root}/scripts/start_full_sft_dense768_e2.sh'
MONITOR_SCRIPT='${monitor_script}'
EOF

trap - EXIT

printf 'STARTED_SCREEN=%s\n' "${session_name}"
printf 'LOG_PATH=%s\n' "${log_path}"
printf 'MONITOR_CSV_PATH=%s\n' "${monitor_csv_path}"
printf 'MANIFEST_PATH=%s\n' "${manifest_path}"
printf 'WRITER_LOCK=%s\n' "${writer_lock}"
printf '\n'
printf 'Writer lock coverage:\n'
printf -- '- This official start script holds advisory flock %s for the full detached train_full_sft.py lifecycle.\n' "${writer_lock}"
printf -- '- Directly bypassing this script and running python train_full_sft.py manually can still bypass the lock.\n'
printf '\n'
printf 'Recommended Windows PowerShell follow-ups:\n'
printf 'wsl -d Ubuntu-24.04 -- cat %s\n' "${manifest_path}"
printf 'wsl -d Ubuntu-24.04 -- screen -ls\n'
printf 'wsl -d Ubuntu-24.04 -- bash -lc "ps -eo pid,ppid,tty,etime,cmd | grep '\''[t]rain_full_sft.py'\''"\n'
printf 'wsl -d Ubuntu-24.04 -- bash %s\n' "${monitor_script}"
