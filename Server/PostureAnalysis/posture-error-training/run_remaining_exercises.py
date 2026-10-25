"""VS Code ▶: Train the remaining four exercises; resume existing pose caches."""
from pathlib import Path
import argparse,os,subprocess,sys
BASE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--exercise',choices=['pushups','elbow_plank','ytw','shoulder_press']);p.add_argument('--workers',type=int,default=12);p.add_argument('--skip-prepare',action='store_true');a=p.parse_args()
    executable=Path(os.environ.get('POSTURE_PYTHON',sys.executable));python=str(executable) if executable.exists() else sys.executable
    env=dict(os.environ,MPLCONFIGDIR='/tmp/posture-mpl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TF_CPP_MIN_LOG_LEVEL='3')
    commands=[]
    if not a.skip_prepare:commands.append(['suite_prepare.py','--workers',str(a.workers)]+(['--only',a.exercise] if a.exercise else []))
    commands.append(['suite_data_audit.py'])
    commands.append(['suite_train.py']+(['--exercise',a.exercise] if a.exercise else []))
    if not a.exercise:commands.append(['test_suite.py'])
    commands+= [['suite_report.py']]
    for command in commands:
        print('Running '+' '.join(command),flush=True);subprocess.run([python,str(BASE/command[0]),*command[1:]],cwd=BASE,env=env,check=True)
    print('Completed. Results: '+str(BASE/'multi/REPORT.md'),flush=True)

if __name__=='__main__':main()
