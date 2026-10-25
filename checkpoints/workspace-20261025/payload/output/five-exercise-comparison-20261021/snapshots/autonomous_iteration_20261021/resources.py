"""영상 처리의 온도·전원·메모리 조건을 검사하여 과열 시 중단한다.

자동 재시작하지 않으며 센서 누락도 실패로 처리한다.
"""

import pathlib
import subprocess

import audit


def check_resources() -> dict:
    """센서·전원·메모리를 검사하고 현재 측정값을 반환한다.

    Returns:
        dict: CPU·GPU 섭씨 온도, 메모리 여유 MiB, 전원 상태.

    Raises:
        RuntimeError: 센서 누락, CPU 85℃·GPU 75℃ 이상, 전원 분리,
            메모리 여유 1500MiB 미만인 경우. 자동 재시작하지 않는다.
        subprocess.SubprocessError: GPU 조회 실패 또는 5초 제한 초과.
    """
    temperatures = []
    for sensor in pathlib.Path("/sys/class/hwmon").glob("hwmon*"):
        if (sensor / "name").read_text().strip() == "coretemp":
            temperatures.extend(
                float(path.read_text()) / 1000
                for path in sensor.glob("temp*_input")
            )
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    cpu_c = max(temperatures) if temperatures else None
    gpu_c = float(gpu.stdout.strip().splitlines()[0])
    available_mib = None
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available_mib = int(line.split()[1]) / 1024
    ac_connected = any(
        path.read_text().strip() == "1"
        for path in pathlib.Path("/sys/class/power_supply").glob("*/online")
    )
    reading = dict(
        cpu_c=cpu_c,
        gpu_c=gpu_c,
        available_mib=available_mib,
        ac_connected=ac_connected,
    )
    audit.save_json(audit.OUT / "resource_status.json", reading)
    if (
        cpu_c is None
        or cpu_c >= 85
        or gpu_c >= 75
        or not ac_connected
        or available_mib is None
        or available_mib < 1500
    ):
        audit.save_json(audit.OUT / "resource_stop.json", reading)
        raise RuntimeError("Resource guard stopped extraction")
    return reading
