#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
manifest_path="${project_root}/experiments/logs/full-sft-current-run.env"
checkpoint_path="${project_root}/checkpoints/full_sft_768_resume.pth"

find_train_pids() {
  ps -eo pid=,comm=,args= | awk '
    $2 ~ /^python([0-9.]*)?$/ && $0 ~ /train_full_sft[.]py/ {
      print $1
    }
  '
}

csv_path="${1:-}"
interval="${2:-30}"

if [[ -z "${csv_path}" ]]; then
  if [[ -f "${manifest_path}" ]]; then
    # shellcheck disable=SC1090
    source "${manifest_path}"
    csv_path="${MONITOR_CSV_PATH:-}"
  fi
fi

if [[ -z "${csv_path}" ]]; then
  printf 'ERROR: missing output CSV path. Pass it explicitly or ensure %s exists.\n' "${manifest_path}" >&2
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

printf 'timestamp,pid,mem_available_kb,swap_used_kb,vmrss_kb,vmswap_kb,rssanon_kb,rssfile_kb,checkpoint_mtime_epoch\n' > "${csv_path}"
printf 'Monitoring pid=%s into %s with interval=%ss\n' "${pid}" "${csv_path}" "${interval}"

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
  mem_available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  swap_total_kb="$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)"
  swap_free_kb="$(awk '/SwapFree:/ {print $2}' /proc/meminfo)"
  swap_used_kb="$((swap_total_kb - swap_free_kb))"
  vmrss_kb="$(awk '/VmRSS:/ {print $2}' "/proc/${pid}/status")"
  vmswap_kb="$(awk '/VmSwap:/ {print $2}' "/proc/${pid}/status")"
  rssanon_kb="$(awk '/RssAnon:/ {print $2}' "/proc/${pid}/status")"
  rssfile_kb="$(awk '/RssFile:/ {print $2}' "/proc/${pid}/status")"

  if [[ -e "${checkpoint_path}" ]]; then
    checkpoint_mtime_epoch="$(stat -c '%Y' "${checkpoint_path}")"
  else
    checkpoint_mtime_epoch=0
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${timestamp}" \
    "${pid}" \
    "${mem_available_kb}" \
    "${swap_used_kb}" \
    "${vmrss_kb}" \
    "${vmswap_kb}" \
    "${rssanon_kb}" \
    "${rssfile_kb}" \
    "${checkpoint_mtime_epoch}" >> "${csv_path}"

  sleep "${interval}"
done
