#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
default_manifest_path="${project_root}/experiments/logs/full-sft-current-run.env"
manifest_path="${MANIFEST_PATH:-${default_manifest_path}}"

find_train_pids() {
  ps -eo pid=,comm=,args= | awk '
    $2 ~ /^python([0-9.]*)?$/ && $0 ~ /train_full_sft[.]py/ {
      print $1
    }
  '
}

read_proc_field_kb() {
  local pid="$1"
  local field="$2"
  awk -v key="${field}" '$1 == key ":" {print $2}' "/proc/${pid}/status"
}

get_screen_state() {
  local session_name="$1"
  if [[ -z "${session_name}" ]]; then
    printf 'unknown'
    return
  fi
  if screen -ls 2>/dev/null | grep -F "${session_name}" >/dev/null; then
    printf 'running'
  else
    printf 'stopped'
  fi
}

get_gpu_pid_mem_mb() {
  local pid="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '0'
    return
  fi
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v target_pid="${pid}" '
      $1 + 0 == target_pid {
        gsub(/ /, "", $2)
        print $2
        found = 1
        exit
      }
      END {
        if (!found) {
          print 0
        }
      }
    '
}

get_gpu_summary() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf 'nvidia-smi unavailable'
    return
  fi
  local summary
  summary="$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{gsub(/^ +| +$/, "", $0); printf "gpu%s %s %s/%s MiB util=%s%%\n", $1, $2, $3, $4, $5}')"
  if [[ -z "${summary}" ]]; then
    printf 'nvidia-smi returned no gpu rows'
  else
    printf '%s' "${summary}"
  fi
}

csv_path="${1:-}"
interval="${2:-30}"

if [[ -f "${manifest_path}" ]]; then
  # shellcheck disable=SC1090
  source "${manifest_path}"
fi

checkpoint_path="${CHECKPOINT_PATH:-${project_root}/checkpoints/full_sft_768_resume.pth}"
screen_session="${SCREEN_SESSION:-}"
log_path="${LOG_PATH:-}"

if [[ -z "${csv_path}" ]]; then
  csv_path="${MONITOR_CSV_PATH:-}"
fi

if [[ -z "${csv_path}" ]]; then
  printf 'ERROR: missing output CSV path. Pass it explicitly or ensure manifest exists: %s\n' "${manifest_path}" >&2
  exit 1
fi

if ! [[ "${interval}" =~ ^[0-9]+$ ]] || [[ "${interval}" -le 0 ]]; then
  printf 'ERROR: interval must be a positive integer, got: %s\n' "${interval}" >&2
  exit 1
fi

mkdir -p "$(dirname -- "${csv_path}")"
if [[ -e "${csv_path}" ]]; then
  printf 'ERROR: refusing to overwrite existing CSV: %s\n' "${csv_path}" >&2
  exit 1
fi

mapfile -t train_pids < <(find_train_pids)
if [[ "${#train_pids[@]}" -eq 0 ]]; then
  printf 'ERROR: no running train_full_sft.py python process found.\n' >&2
  exit 1
fi
if [[ "${#train_pids[@]}" -ne 1 ]]; then
  printf 'ERROR: expected exactly one train_full_sft.py python process, found: %s\n' "${train_pids[*]}" >&2
  exit 1
fi

pid="${train_pids[0]}"

printf 'timestamp,pid,screen_alive,mem_available_kb,swap_used_kb,vmrss_kb,vmswap_kb,rssanon_kb,rssfile_kb,gpu_pid_mem_mb,checkpoint_mtime_epoch\n' > "${csv_path}"
printf 'Monitoring pid=%s into %s with interval=%ss\n' "${pid}" "${csv_path}" "${interval}"
printf 'Manifest path: %s\n' "${manifest_path}"
printf 'Screen session: %s\n' "${screen_session:-<unset>}"
printf 'Log path: %s\n' "${log_path:-<unset>}"
printf 'Checkpoint path: %s\n' "${checkpoint_path}"

while true; do
  if ! kill -0 "${pid}" 2>/dev/null; then
    printf 'train_full_sft.py pid=%s has exited; monitor stopping normally.\n' "${pid}"
    exit 0
  fi

  if [[ ! -r "/proc/${pid}/status" ]]; then
    printf 'train_full_sft.py pid=%s status file is no longer readable; monitor stopping.\n' "${pid}"
    exit 0
  fi

  timestamp="$(date --iso-8601=seconds)"
  screen_state="$(get_screen_state "${screen_session}")"
  mem_total_kb="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
  mem_available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  swap_total_kb="$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)"
  swap_free_kb="$(awk '/SwapFree:/ {print $2}' /proc/meminfo)"
  swap_used_kb="$((swap_total_kb - swap_free_kb))"
  vmrss_kb="$(read_proc_field_kb "${pid}" "VmRSS")"
  vmswap_kb="$(read_proc_field_kb "${pid}" "VmSwap")"
  rssanon_kb="$(read_proc_field_kb "${pid}" "RssAnon")"
  rssfile_kb="$(read_proc_field_kb "${pid}" "RssFile")"
  gpu_pid_mem_mb="$(get_gpu_pid_mem_mb "${pid}")"
  gpu_summary="$(get_gpu_summary)"
  process_line="$(ps -p "${pid}" -o pid=,ppid=,etime=,%cpu=,%mem=,cmd=)"

  if [[ -e "${checkpoint_path}" ]]; then
    checkpoint_mtime_epoch="$(stat -c '%Y' "${checkpoint_path}")"
  else
    checkpoint_mtime_epoch=0
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${timestamp}" \
    "${pid}" \
    "${screen_state}" \
    "${mem_available_kb}" \
    "${swap_used_kb}" \
    "${vmrss_kb}" \
    "${vmswap_kb}" \
    "${rssanon_kb}" \
    "${rssfile_kb}" \
    "${gpu_pid_mem_mb}" \
    "${checkpoint_mtime_epoch}" >> "${csv_path}"

  printf '\n[%s]\n' "${timestamp}"
  printf 'screen: %s (%s)\n' "${screen_session:-<unset>}" "${screen_state}"
  printf 'python: %s\n' "${process_line}"
  printf 'gpu(pid): pid=%s gpu_mem=%s MiB\n' "${pid}" "${gpu_pid_mem_mb}"
  printf 'gpu(summary):\n%s\n' "${gpu_summary}"
  printf 'wsl-mem: total=%s kB available=%s kB\n' "${mem_total_kb}" "${mem_available_kb}"
  printf 'swap: used=%s kB total=%s kB free=%s kB\n' "${swap_used_kb}" "${swap_total_kb}" "${swap_free_kb}"
  printf 'proc-mem: VmRSS=%s kB VmSwap=%s kB RssAnon=%s kB RssFile=%s kB\n' \
    "${vmrss_kb}" \
    "${vmswap_kb}" \
    "${rssanon_kb}" \
    "${rssfile_kb}"
  printf 'checkpoint: %s (mtime_epoch=%s)\n' "${checkpoint_path}" "${checkpoint_mtime_epoch}"
  printf 'csv: %s\n' "${csv_path}"

  if [[ -n "${log_path}" && -f "${log_path}" ]]; then
    printf 'log-tail:\n'
    tail -n 20 "${log_path}"
  else
    printf 'log-tail: unavailable\n'
  fi

  sleep "${interval}"
done
