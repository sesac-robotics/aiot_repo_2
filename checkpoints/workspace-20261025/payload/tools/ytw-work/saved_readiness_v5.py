"""Unlabelled saved pose diagnostic. Not live activation or a true-count test."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np,joblib
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/ytw-work-20261024/readiness-v5-20261025';A=B/'output/live-webcam-20261023';S=A/'runtime/c76334937f21ef24a27aa1c1687de6c9577bf6249e5cac606151a328893746f7';sys.path[:0]=[str(A),str(S)]
from ytw_candidate import LearnedPhaseSource
from ytw_hold_onset_v4 import YTWTracker as Parent
from ytw_readiness_v5 import YTWTracker,scoring_input_hold
from app.scoring.initial_pose_return import pose
from live_observation import mask_frame_border
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();cp=A/'sessions/1792846426614521826/pose.npz';ch=sha(cp);z=np.load(cp);w,h=int(z['width']),int(z['height']);xy=mask_frame_border(z['image_xy'],w,h);bp=B/'output/initial-pose-return-v7-20261024/initial-pose-return-bundle.joblib';bundlehash=sha(bp);assert bundlehash=='fc18ccb7f3ab9a8a9b23d1683be07de83530ffebcf657046b81d69549a6e0439';bundle=joblib.load(bp);LearnedPhaseSource(bundle)
v=[pose('ytw',p) for p in xy];ii=[i for i,a in enumerate(v) if a is not None];X=[v[i] for i in ii];phase=bundle['heads'][('ytw','x','phase')]['model'].predict(X);ready=bundle['heads'][('ytw','x','target')]['model'].predict(X);pred={i:(str(a),bool(b==1)) for i,a,b in zip(ii,phase,ready)}
outputs=[]
for name,cls in [('parent_v4.1',Parent),('readiness_v5',YTWTracker)]:
 current=[0];t=cls(lambda p:pred.get(current[0],(None,False)));events=[];states=[]
 for i,(tick,p) in enumerate(zip(z['frame_numbers'],z['image_xy'])):
  current[0]=i;e=t.push(int(tick),p,w,h)
  if e:events.append(e)
  if name=='readiness_v5':states.append(dict(tick=int(tick),state=t.status()))
 outputs.append(dict(candidate=name,events=events,statuses=states))
a,b=outputs
assert len(a['events'])==len(b['events'])==1
for key in ['start_tick','end_tick','auto_phase_windows','observation_coverage']:assert a['events'][0][key]==b['events'][0][key]
assert np.allclose([p for t,p in a['events'][0]['samples']],[p for t,p in b['events'][0]['samples']],equal_nan=True)
assert scoring_input_hold(b['events'][0])['input_contract_eligible']
for state in b['statuses']:
 if not state['state']['current_observation']['valid']:assert state['state']['current_Y_ready'] is False
for output in outputs:output['events']=[{k:v for k,v in e.items() if k!='samples'} for e in output['events']]
assert sha(cp)==ch and sha(bp)==bundlehash
result=dict(schema='ytw-readiness-v5-unlabelled-saved-replay',scope='no_video_or_quality_truth_not_independent_accuracy',source_sha256=ch,state_bundle_sha256=bundlehash,state_bundle_activated=False,summary=dict(samples=len(xy),unobserved_frames=sum(not s['state']['current_observation']['valid'] for s in b['statuses']),ready_on_unobserved_frames=0,events=1,segmentation_or_score_input_changed=False,state_counts=dict(Counter(s['state']['current_state'] for s in b['statuses']))),results=outputs,candidate_code_sha256=sha(B/'tools/ytw-work/ytw_readiness_v5.py'))
(O/'saved-session.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result['summary'],ensure_ascii=False,indent=2))
