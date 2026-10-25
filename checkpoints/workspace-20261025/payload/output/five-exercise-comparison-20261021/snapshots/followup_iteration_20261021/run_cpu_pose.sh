#!/usr/bin/env bash
# 역할: 새 Y 후보 검토·최대 두 CPU Heavy 구간 또는 최대9개 표시 영상만 실행한다.
# GPU 실행은 금지하고 기존 GPU AC 보호를 변경하지 않는다.
# 입력: prepare, extract, render. 시작 잔량85%, 중단 잔량80%, CPU70/GPU60도.
# 출력: 단일 실행 로그. 실패 시 자동 재시작하지 않는다.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^(prepare|extract|render)$ ]]; then exit 2; fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
base_dir="$(dirname -- "$script_dir")"
output_dir="$base_dir/output/five-exercise-comparison-20261021"
cd -- "$script_dir"
"$base_dir/aiot_repo_2/R2D2/bin/python" -c 'from cpu_analysis_guard import check_resources; r=check_resources("short_heavy_cpu"); assert r["cpu_c"]<55 and r["gpu_c"]<50 and (r["ac_connected"] or r["battery_percent"]>=85), "CPU candidate start guard"'
unit="five-exercise-cpu-side-$(date +%Y%m%d-%H%M%S)"
exec systemd-run --user --wait --collect --unit="$unit" \
  --property=MemoryMax=1500M --property=CPUQuota=25% \
  --property=CPUQuotaPeriodSec=10ms --property=RuntimeMaxSec=300 \
  --property="StandardOutput=append:$output_dir/$unit.log" \
  --property="StandardError=append:$output_dir/$unit.log" \
  --working-directory="$base_dir" \
  /usr/bin/flock --nonblock "$output_dir/video-processing.lock" \
  /usr/bin/env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 \
  "$base_dir/aiot_repo_2/R2D2/bin/python" "$script_dir/followup_y_side_cpu.py" "$1"
