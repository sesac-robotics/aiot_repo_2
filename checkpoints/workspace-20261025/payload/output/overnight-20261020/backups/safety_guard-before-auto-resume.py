"""야간 작업의 온도·전원을 약 10~20초마다 검사한다. 화재 감지기나 하드웨어 안전 보증은 아니다."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import signal

OUT=Path('/home/jayden/SeSACPAI-Mid/output/overnight-20261020')
UNIT='posture-overnight-20261020.service'


def write(path,value):
    """검사 결과를 UTF-8 JSON으로 원자 저장한다. 저장 실패는 호출자에게 전달한다."""
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)


def snapshot():
    """CPU·GPU 섭씨 온도, 전원 연결 여부, 배터리 백분율을 반환한다."""
    cpu=[]
    for p in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:
            if (p/'type').read_text().strip() in ['x86_pkg_temp','TCPU_PCI','acpitz']:
                cpu.append(int((p/'temp').read_text())/1000)
        except (OSError,ValueError):pass
    for d in Path('/sys/class/hwmon').glob('hwmon*'):
        try:
            if (d/'name').read_text().strip()=='coretemp':
                for f in d.glob('temp*_input'):cpu.append(int(f.read_text())/1000)
        except (OSError,ValueError):pass
    gpu=None
    result=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
    try:gpu=float(result.stdout.strip().splitlines()[0])
    except (ValueError,IndexError):pass
    supplies=list(Path('/sys/class/power_supply').glob('*'))
    ac=any((p/'online').exists() and (p/'online').read_text().strip()=='1' for p in supplies)
    battery=[int((p/'capacity').read_text()) for p in supplies if (p/'capacity').exists()]
    return dict(time=datetime.datetime.now().astimezone().isoformat(),cpu_c=max(cpu) if cpu else None,gpu_c=gpu,ac_connected=ac,battery_percent=min(battery) if battery else None)


def stop_work(reason,reading):
    """결과 보존 상태를 기록한 후 작업 서비스만 명시적으로 중단해 자동 재시작을 막는다."""
    value=dict(reason=reason,reading=reading,action='작업 서비스 중단. 저장된 결과 유지. 조건 확인 후 수동 재개 필요.',notice='이 임계값은 보수적인 작업 중단 기준이며 화재 위험 임계값이 아님.')
    write(OUT/'safety_stop.json',value)
    subprocess.run(['systemctl','--user','stop',UNIT],check=True,timeout=45)
    env=os.environ.copy();env.setdefault('DBUS_SESSION_BUS_ADDRESS','unix:path=/run/user/1000/bus')
    subprocess.run(['notify-send','--urgency=critical','--expire-time=0','야간 학습을 보호 중단했습니다',reason],env=env,timeout=10)


def thaw():
    """작업 서비스만 실행 상태로 돌린다. 다른 앱·GPU 전역 설정은 변경하지 않는다."""
    subprocess.run(['systemctl','--user','thaw',UNIT],capture_output=True,timeout=10)


def cool_rest(seconds):
    """작업 프로세스 묶음에만 짧은 냉각 휴식을 주고 반드시 재개한다.

    Args:
        seconds: 휴식 시간, 초.
    Raises:
        CalledProcessError: 서비스 일시 정지 실패.
    """
    subprocess.run(['systemctl','--user','freeze',UNIT],check=True,timeout=10)
    try:time.sleep(seconds)
    finally:thaw()


def main():
    """정상 온도는 전속력으로 처리하고 지속 고온에만 단계적으로 부하를 줄인다."""
    signal.signal(signal.SIGTERM,lambda signum,frame: (_ for _ in ()).throw(SystemExit(0)))
    hot_since=None;critical_since=None;cool_since=None;mode='normal'
    thaw()
    try:
        while True:
            result=subprocess.run(['systemctl','--user','show',UNIT,'-p','ActiveState','--value'],capture_output=True,text=True,timeout=8)
            if result.stdout.strip() not in ['active','activating','reloading']:return
            reading=snapshot();now=time.monotonic()
            cpu=reading['cpu_c'] or 0;gpu=reading['gpu_c'] or 0
            hot=cpu>=95 or gpu>=88
            critical=cpu>=99 or gpu>=95
            hot_since=(hot_since if hot_since is not None else now) if hot else None
            critical_since=(critical_since if critical_since is not None else now) if critical else None
            reason=None
            if cpu>=100 or gpu>=97:reason='CPU 100℃ 또는 GPU 97℃ 도달로 작업 보호 중단'
            elif critical_since is not None and now-critical_since>=20:reason='CPU 99℃ 또는 GPU 95℃ 이상이 20초 지속되어 작업 보호 중단'
            elif not reading['ac_connected'] and reading['battery_percent'] is not None and reading['battery_percent']<=25:reason='충전기 미연결 상태에서 배터리 25% 이하'
            if reason:
                thaw();stop_work(reason,reading);return
            if hot_since is not None and now-hot_since>=60:
                mode='cooling_50_percent' if now-hot_since>=180 else 'cooling_25_percent'
            cool=cpu<=92 and gpu<=85
            cool_since=(cool_since if cool_since is not None else now) if cool else None
            if mode!='normal' and cool_since is not None and now-cool_since>=20:mode='normal';hot_since=None
            reading.update(mode=mode,hot_seconds=0 if hot_since is None else round(now-hot_since),policy='평소 전속력. 지속 고온에만 작업 프로세스의 실행 시간을 줄임. 화재 위험 기준이 아님.')
            write(OUT/'safety_status.json',reading)
            if mode=='cooling_25_percent':
                # 15초 가동·5초 휴식: 정상 상태에는 이 제한을 적용하지 않는다.
                cool_rest(5);time.sleep(15)
            elif mode=='cooling_50_percent':
                cool_rest(10);time.sleep(10)
            else:time.sleep(10)
    finally:thaw()


if __name__=='__main__':main()
