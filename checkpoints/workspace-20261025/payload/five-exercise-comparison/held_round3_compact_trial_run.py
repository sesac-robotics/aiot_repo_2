"""보호 실행기에서 별도 환경의 학습·개발 비교를 수행한다."""
import os
import subprocess

import audit
from resources import check_resources


def main() -> None:
    """온도·메모리 확인 후 기존 서비스 환경을 바꾸지 않고 실행한다."""
    check_resources()
    env=dict(os.environ)
    for key in ('LD_PRELOAD','PYTHONPATH','__EGL_VENDOR_LIBRARY_FILENAMES'):env.pop(key,None)
    script=audit.BASE/'output/five-exercise-recovery-20261023/plank_compact_trial.py'
    subprocess.run([str(audit.BASE/'.posture-score-verify/bin/python'),'-c',f"import runpy; runpy.run_path({str(script)!r},run_name='__main__')"],cwd=audit.BASE/'team-posture-scoring/Server/AIService',env=env,check=True,timeout=1400)


if __name__=='__main__':main()
