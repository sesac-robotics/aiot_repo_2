from pathlib import Path
import json,sys,os,subprocess
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-rebuild-v2-20261025';sys.path.insert(0,str(B/'five-exercise-comparison'));import extract
plan=json.loads((O/'plan.frozen.json').read_text())['rows'];receipts=[]
def resources():
 temp=[]
 for p in Path('/sys/class/hwmon').glob('hwmon*'):
  if (p/'name').read_text().strip()=='coretemp':temp.extend(float(f.read_text())/1000 for f in p.glob('temp*_input'))
 available=next(int(s.split()[1])/1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
 if not temp or max(temp)>=85 or available<1500:raise RuntimeError('CPU temperature or memory guard')
 return {'cpu_c':max(temp),'available_mib':available}
for r in plan:
 if not r['requires_extraction']:continue
 # Include actual warmup observations, but only the declared event context can become training input.
 fps=r['fps'];start=max(0,r['context_seconds'][0]-2);end=r['context_seconds'][1]
 row=dict(id=r['id'],video_sha256=r['source_sha256'],required_views=['front'],segment=dict(start_frame=int(start*fps),end_frame=int(end*fps),fps=fps))
 z=extract.extract_view(row,'primary',r['path'],O/'observations',delegate='CPU',resource_checker=resources)
 receipts.append(dict(id=r['id'],cache=z['cache'],frames=z['frames'],delegate='CPU'));(O/'extraction-progress.json').write_text(json.dumps(receipts,indent=2)+'\n');print(r['id'],z['frames'],flush=True)
print('finished',len(receipts),flush=True)
