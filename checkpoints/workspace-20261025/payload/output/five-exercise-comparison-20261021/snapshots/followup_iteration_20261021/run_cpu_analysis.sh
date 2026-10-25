#!/usr/bin/env bash
# 역할: 허용된 저장 관절 통계·학습·HTML 다섯 작업만 CPU25%로 실행한다.
# 입력: 아래 허용 목록의 Python 파일 이름 하나. 출력: 보호된 실행 로그.
# 실패: 허용 목록 밖 파일, 시작 온도 CPU55/GPU50도 이상, 배터리75% 미만,
#       메모리 부족 또는 실행 오류. GPU 추출용 AC 필수 설정은 바꾸지 않는다.
set -euo pipefail
if [[ $# -ne 1 ]]; then exit 2; fi
case "$1" in
    plank_monotone_trial.py|evaluate_first_cycle_trial.py|plank_world_trial.py|render_followup_results.py|plank_masked_multitask_trial.py) ;;
    *) echo 'Only the five cached-data/statistics/report scripts are permitted.' >&2; exit 2 ;;
esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
base_dir="$(dirname -- "$script_dir")"
output_dir="$base_dir/output/five-exercise-comparison-20261021"
cd -- "$script_dir"
"$base_dir/aiot_repo_2/R2D2/bin/python" -c 'from cpu_analysis_guard import check_resources; r=check_resources(); assert r["cpu_c"]<55 and r["gpu_c"]<50, "Cool before cached analysis"'
unit="five-exercise-cpu-analysis-$(date +%Y%m%d-%H%M%S)"
exec systemd-run --user --wait --collect --unit="$unit" \
  --property=MemoryMax=1G --property=CPUQuota=25% \
  --property=CPUQuotaPeriodSec=10ms --property=RuntimeMaxSec=180 \
  --property="StandardOutput=append:$output_dir/$unit.log" \
  --property="StandardError=append:$output_dir/$unit.log" \
  --working-directory="$base_dir" \
  /usr/bin/flock --nonblock "$output_dir/video-processing.lock" \
  /usr/bin/env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 \
  "$base_dir/aiot_repo_2/R2D2/bin/python" "$script_dir/$1"
