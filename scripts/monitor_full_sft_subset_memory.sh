#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"

export MANIFEST_PATH="${project_root}/experiments/logs/full-sft-subset-current-run.env"

exec bash "${project_root}/scripts/monitor_full_sft_memory.sh" "$@"
