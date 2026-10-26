#!/usr/bin/env bash
# 역할: 로컬 실험 Python 작업을 저부하 서비스와 단일 실행 잠금으로 실행한다.
# 입력: 영문 작업 이름, 이 폴더의 Python 파일 이름, 선택 실행 인수.
# 출력: 서비스 식별자와 output 폴더의 작업별 로그. 온도 단위는 섭씨다.
# 실패: CPU 70도·GPU 65도 이상에서 시작 거부, 실행 스크립트 실패,
#       잠금 충돌, 유효하지 않은 입력, systemd 또는 Python 실행 실패.
# 실행 중 센서 검사는 extract·external_review·render·verify 내부에서 수행한다.
# 자동 재시작하지 않는다. Git 저장소·시스템 전원 설정은 변경하지 않는다.
set -euo pipefail
if [[ $# -lt 2 || ! "$1" =~ ^[a-z][a-z0-9-]{0,48}$ ]]; then
    echo 'Usage: run_guarded.sh job-name local-script.py [arguments...]' >&2
    exit 2
fi
job_name="$1"
script_name="$2"
shift 2
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
base_dir="$(dirname -- "$script_dir")"
output_dir="$base_dir/output/five-exercise-comparison-20261021"
if [[ "$script_name" == */* || "$script_name" != *.py || ! -f "$script_dir/$script_name" ]]; then
    echo 'Only Python files in this experiment folder are accepted.' >&2
    exit 2
fi
cd -- "$script_dir"
"$base_dir/aiot_repo_2/R2D2/bin/python" -c 'from resources import check_resources; r=check_resources(); assert r["cpu_c"] < 70 and r["gpu_c"] < 65, "Cool before starting another job"'
unit_name="five-exercise-$job_name-$(date +%Y%m%d-%H%M%S)"
exec systemd-run --user --wait --collect --unit="$unit_name" \
    --property=MemoryMax=3G --property=CPUQuota=50% \
    --property=CPUQuotaPeriodSec=10ms --property=RuntimeMaxSec=2400 \
    --property="StandardOutput=append:$output_dir/$unit_name.log" \
    --property="StandardError=append:$output_dir/$unit_name.log" \
    --working-directory="$base_dir" \
    /usr/bin/flock --nonblock "$output_dir/video-processing.lock" \
    /usr/bin/env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    "$base_dir/mediapipe-pilot/run_nvidia.sh" "$script_dir/$script_name" "$@"
