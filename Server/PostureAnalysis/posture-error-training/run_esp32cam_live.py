"""이 파일을 열고 VS Code 오른쪽 위 ▶를 누르세요.
ESP32-CAM 실시간 영상 + 관절 표시 + 학습 모델의 오류/교정 문구를 엽니다.
"""
from pathlib import Path
import os,sys

def main():
    runtime=Path(os.environ.get('POSTURE_PYTHON',sys.executable))
    env=dict(os.environ,MPLCONFIGDIR='/tmp/posture-mpl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TF_CPP_MIN_LOG_LEVEL='3')
    if runtime.exists() and runtime.resolve()!=Path(sys.executable).resolve():os.execve(str(runtime),[str(runtime),str(Path(__file__).resolve()),*sys.argv[1:]],env)
    from live_app.server import main as run
    run()

if __name__=='__main__':main()
