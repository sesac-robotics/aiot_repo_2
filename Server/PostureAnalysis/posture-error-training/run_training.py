"""VS Code ▶: extract squat poses, audit duplicates, train and evaluate CNN/GRU."""
from pathlib import Path
import os,subprocess,sys
BASE=Path(__file__).resolve().parent
PYTHON=Path(os.environ.get('POSTURE_PYTHON',sys.executable))

def main():
    python=str(PYTHON) if PYTHON.exists() else sys.executable
    env=dict(os.environ,MPLCONFIGDIR='/tmp/posture-mpl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TF_CPP_MIN_LOG_LEVEL='3')
    for script in ['prepare.py','audit_duplicates.py','train.py','test_pipeline.py','make_report.py']:
        print(f'Running {script}',flush=True)
        subprocess.run([python,str(BASE/script)],cwd=BASE,env=env,check=True)
    print(f'Results: {BASE / "outputs/REPORT.md"}',flush=True)

if __name__=='__main__':main()
