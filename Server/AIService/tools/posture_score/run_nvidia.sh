#!/usr/bin/env bash
# 역할: 이번 실행 프로세스에서만 NVIDIA EGL 장치를 지정한다.
# 입력: 사용할 Python 실행 파일과 -m app.scoring 이후의 CLI 인수.
# 출력: GPU 추출 결과. 컴파일된 어댑터는 사용자 캐시에만 저장한다.
# 실패: 드라이버·gcc·Python·GPU 초기화 실패. CPU로 자동 대체하지 않는다.
set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo 'Usage: run_nvidia.sh /path/to/python [scoring arguments]' >&2
    exit 2
fi
python_bin="$1"
shift
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/sesac-posture-score"
library="$cache_dir/nvidia_egl_device.so"
vendor=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
[[ -x "$python_bin" && -r "$vendor" ]]
mkdir -p -- "$cache_dir"
if [[ ! -f "$library" || "$script_dir/nvidia_egl_device.c" -nt "$library" ]]; then
    gcc -shared -fPIC -Wall -Wextra -o "$library.tmp" \
        "$script_dir/nvidia_egl_device.c" -ldl
    mv -- "$library.tmp" "$library"
fi
exec env __EGL_VENDOR_LIBRARY_FILENAMES="$vendor" \
    LD_PRELOAD="$library${LD_PRELOAD:+:$LD_PRELOAD}" \
    "$python_bin" -m app.scoring "$@" --delegate GPU
