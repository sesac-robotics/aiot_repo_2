from pathlib import Path
import sys,json,hashlib
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid'); L=B/'output/live-webcam-20261023';O=B/'output/shoulder-work-20261024'
sys.path.insert(0,str(L))
from shoulder_candidate import ShoulderTracker
from movement import MovementTracker
from test_shoulder_candidate import pose,replay
S=L/'sessions/1792846279497155109';d=json.loads((S/'final-state.json').read_text());p=np.load(S/'pose.npz')
rows={}
for name,tracker in [('patched_shared',MovementTracker('shoulder')),('candidate',ShoulderTracker())]:
    for tick,xy in zip(p['frame_numbers'],p['image_xy']):tracker.push(int(tick),xy,int(p['width']),int(p['height']))
    rows[name]=tracker.status()
runtime=L/'runtime'/d['loaded_code_model_sha256'];sys.path.insert(0,str(runtime))
from app.scoring.engine import ScoringEngine
from app.scoring.sequence import PoseSequence
engine=ScoringEngine()
t,segments=replay([pose()]*6+[pose(True)]*4+[pose()]*3)
seg=segments[0];ticks=np.array([a for a,_ in seg]);xy=np.array([b for _,b in seg])
result=engine.score('shoulder',PoseSequence(xy,ticks-ticks[0],10.,640,480),view='front',complete_attempt=True)
def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    return x
report={'scope':'functional replay and synthetic pipeline; no accuracy claim','saved_count':d['movement']['count'],'saved_repetitions':d['repetitions'],'replay':rows,'synthetic_scoring':result,'runtime_digest':d['loaded_code_model_sha256'],'source_pose_sha256':hashlib.sha256((S/'pose.npz').read_bytes()).hexdigest(),'training_performed':False,'independent_final_test_opened':False}
(O/'verification.json').write_text(json.dumps(safe(report),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(safe({'replay_counts':{k:v['count'] for k,v in rows.items()},'score':result}),ensure_ascii=False))
