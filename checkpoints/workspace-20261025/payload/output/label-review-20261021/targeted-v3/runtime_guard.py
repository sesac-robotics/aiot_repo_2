"""짧은 관절 추출·표시 작업의 온도 점검을 공통으로 제공한다."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path('/home/jayden/SeSACPAI-Mid/overnight-posture-pipeline')))
import safety_guard


def temperature_check():
    """온도 센서를 확인하고 실행 가능한 상태를 반환한다.

    Returns:
        dict: CPU·GPU 온도(℃), 전원 상태 등의 관측값.
    Raises:
        RuntimeError: 센서가 없거나 CPU 95℃·GPU 85℃ 이상인 경우.
    """
    reading=safety_guard.snapshot()
    if reading['cpu_c'] is None or reading['gpu_c'] is None or reading['cpu_c']>=95 or reading['gpu_c']>=85:raise RuntimeError('고온 또는 온도 센서 오류로 분석을 중단했습니다.')
    return reading
