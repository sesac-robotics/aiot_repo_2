#!/usr/bin/env bash
# 역할: 점수 전용 가상환경을 새로 만들고 고정 의존성을 설치한다.
# 입력: [새 가상환경 경로] [core|video]. PYTHON_BIN은 사용할 Python 경로다.
# 출력: 설치 환경과 실행 예시. 기존 환경·기존 서비스 요구사항은 변경하지 않는다.
# 실패: 기존 대상 경로, venv·pip·바이너리 wheel 설치 실패 시 종료한다.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
component_dir="$(cd -- "$script_dir/../.." && pwd)"
runtime_dir="${1:-$component_dir/.venv}"
mode="${2:-core}"
python_bin="${PYTHON_BIN:-python3}"
if [[ -e "$runtime_dir" || ( "$mode" != core && "$mode" != video ) ]]; then
    echo 'Choose a new environment path and core or video mode.' >&2
    exit 2
fi
"$python_bin" -m pip --version >/dev/null
"$python_bin" -m venv --without-pip "$runtime_dir"
"$python_bin" -m pip --python "$runtime_dir/bin/python" install --only-binary=:all: \
    pip==26.2.1 \
    -r "$component_dir/requirements-posture-score.txt"
if [[ "$mode" == video ]]; then
    "$python_bin" -m pip --python "$runtime_dir/bin/python" install --only-binary=:all: \
        -r "$component_dir/requirements-posture-video.txt"
fi
printf 'Verified runtime: Python 3.14 on Linux. Environment: %s\n' "$runtime_dir"
printf 'From Server/AIService, run: %s/bin/python -m app.scoring --help\n' "$runtime_dir"
