#!/usr/bin/env bash
# 역할: 이 명령의 Python 프로세스에서만 NVIDIA EGL 장치를 선택한다.
# 입력: Python 스크립트 경로 또는 -c/-m 옵션과 해당 인수.
# 출력: 실행한 Python의 표준 출력·오류 및 종료 코드. 물리 단위 없음.
# 실패: 컴파일러·드라이버·Python이 없거나 EGL 초기화·추론에 실패하면 종료한다.
set -euo pipefail
pilot_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$pilot_dir/../aiot_repo_2/R2D2/bin/python"
egl_source="$pilot_dir/runtime/nvidia_egl_device.c"
egl_library="$pilot_dir/runtime/libnvidia_egl_device.so"
egl_vendor=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
if [[ $# -eq 0 ]]; then
    echo '사용법: run_nvidia.sh <Python 스크립트> [인수...]' >&2
    exit 2
fi
[[ -x "$python_bin" && -r "$egl_vendor" ]]
if [[ ! -f "$egl_library" || "$egl_source" -nt "$egl_library" ]]; then
    gcc -shared -fPIC -Wall -Wextra -o "$egl_library" "$egl_source" -ldl
fi
# NVIDIA를 사용할 수 없을 때 Intel로 조용히 대체하지 않는다.
exec env __EGL_VENDOR_LIBRARY_FILENAMES="$egl_vendor" \
    LD_PRELOAD="$egl_library${LD_PRELOAD:+:$LD_PRELOAD}" \
    "$python_bin" "$@"
