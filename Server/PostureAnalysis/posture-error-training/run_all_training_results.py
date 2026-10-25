"""VS Code ▶: show saved five-exercise model results without retraining."""
from pathlib import Path
import os,sys
BASE=Path(__file__).resolve().parent

def main():
    runtime=Path(os.environ.get('POSTURE_PYTHON',sys.executable))
    env=dict(os.environ,MPLCONFIGDIR=os.environ.get('MPLCONFIGDIR','/tmp/posture-mpl'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    if runtime.exists() and runtime.resolve()!=Path(sys.executable).resolve():
        os.execve(str(runtime),[str(runtime),str(Path(__file__).resolve()),*sys.argv[1:]],env)
    from suite_report import main as build_report
    sys.path.insert(0,str(BASE.parent/'shoulder-qevd'))
    from run_demo import main as run
    build_report()
    run(default_page='/all_training_results.html',model_results=True)

if __name__=='__main__':main()
