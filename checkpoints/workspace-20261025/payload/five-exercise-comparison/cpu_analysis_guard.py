"""저장 관절의 통계 분석 전용 저부하 자원 조건을 검사한다.

GPU 추출용 resources.py의 AC 필수 조건은 변경하지 않는다. 기본 프로필은
허용 목록의 캐시 통계·항목 학습·HTML 다섯 작업이다. 별도 단기 CPU 프로필은 새 측면
후보 한 원본과 4초 이하 표시 구간만 허용하는 실행기에서 사용한다.
"""

import pathlib
import subprocess

import audit


def check_resources(scope: str = "cached_statistics") -> dict:
    """CPU 통계 작업의 온도·메모리·잔량을 확인하고 중단 조건을 적용한다.

    Args:
        scope: cached_statistics 또는 short_heavy_cpu. 후자는 별도 허용된
            짧은 CPU 추출 작업이며 배터리 중단 기준을 80%로 높인다.

    Returns:
        dict: CPU·GPU 온도(℃), 메모리(MiB), AC 연결 및 배터리 잔량(%).
        기본 캐시는 배터리 75%, 명시적 단기 CPU 작업은 80% 이상을 요구한다.

    Raises:
        RuntimeError: CPU 70℃·GPU 60℃ 이상, 메모리 2500MiB 미만,
            센서 누락 또는 AC 미연결 상태에서 배터리 75% 미만.
        subprocess.SubprocessError: GPU 온도 조회 실패.
        ValueError: 허용하지 않은 CPU 프로필 이름.
    """
    if scope not in ("cached_statistics", "short_heavy_cpu"):
        raise ValueError("Unknown low-power CPU scope")
    temperatures = []
    for sensor in pathlib.Path("/sys/class/hwmon").glob("hwmon*"):
        if (sensor / "name").read_text().strip() == "coretemp":
            temperatures.extend(
                float(p.read_text()) / 1000 for p in sensor.glob("temp*_input")
            )
    cpu = max(temperatures) if temperatures else None
    gpu = float(
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        .strip()
        .splitlines()[0]
    )
    power = pathlib.Path("/sys/class/power_supply")
    ac = any(p.read_text().strip() == "1" for p in power.glob("*/online"))
    capacity = [
        int((s / "capacity").read_text())
        for s in power.iterdir()
        if (s / "type").read_text().strip() == "Battery"
        and (s / "capacity").exists()
    ]
    battery = min(capacity) if capacity else None
    available = next(
        (
            int(line.split()[1]) / 1024
            for line in pathlib.Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        ),
        None,
    )
    reading = dict(
        cpu_c=cpu,
        gpu_c=gpu,
        available_mib=available,
        ac_connected=ac,
        battery_percent=battery,
        scope="CPU25_" + scope + "_no_GPU_inference",
    )
    audit.save_json(audit.OUT / "cpu_analysis_resource_status.json", reading)
    if (
        cpu is None
        or cpu >= 70
        or gpu >= 60
        or available is None
        or available < 2500
        or (
            not ac
            and (
                battery is None
                or battery < (80 if scope == "short_heavy_cpu" else 75)
            )
        )
    ):
        audit.save_json(audit.OUT / "cpu_analysis_resource_stop.json", reading)
        raise RuntimeError("CPU analysis guard stopped cached statistics")
    return reading
