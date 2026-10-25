"""Unlabelled live-session diagnostic replay, never a quality/true-count test."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,json,hashlib,importlib.util
from pathlib import Path
import numpy as np,joblib
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/ytw-work-20261024/y-entry-v4.1';A=B/'output/live-webcam-20261023';S=A/'runtime/c76334937f21ef24a27aa1c1687de6c9577bf6249e5cac606151a328893746f7';sys.path[:0]=[str(A),str(S)]
from ytw_candidate import LearnedPhaseSource,score_event
from ytw_ready_fix_v2 import YTWTracker as Fixed
from ytw_hold_onset_v4 import YTWTracker as Refined
from app.scoring.initial_pose_return import pose
from app.scoring.engine import ScoringEngine
from live_observation import mask_frame_border
spec=importlib.util.spec_from_file_location('ytw_saved_v1',B/'output/ytw-work-20261024/y-entry-v3/baseline_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();p=A/'sessions/1792846426614521826/pose.npz';original=sha(p);z=np.load(p);w,h=int(z['width']),int(z['height']);xy=mask_frame_border(z['image_xy'],w,h);ticks=z['frame_numbers'];bp=B/'output/initial-pose-return-v7-20261024/initial-pose-return-bundle.joblib';bundle=joblib.load(bp);gate=LearnedPhaseSource(bundle);v=[pose('ytw',a) for a in xy];ii=[i for i,x in enumerate(v) if x is not None];X=[v[i] for i in ii];phase=bundle['heads'][('ytw','x','phase')]['model'].predict(X);ready=bundle['heads'][('ytw','x','target')]['model'].predict(X);trace={i:(str(a),bool(b==1)) for i,a,b in zip(ii,phase,ready)};engine=ScoringEngine();out=[]
for name,cls in [('baseline',m.YTWTracker),('ready_origin_fix',Fixed),('hold_onset_v4.1',Refined)]:
 current=[0];tracker=cls(lambda p:trace.get(current[0],(None,False)));events=[]
 for i,(t,a) in enumerate(zip(ticks,xy)):
  current[0]=i;e=tracker.push(int(t),a,w,h)
  if e:
   ev={k:v for k,v in e.items() if k!='samples'}
   if e['auto_phase_windows'] is not None:ev['same_dense_AI_Rule_result']=score_event(engine,e)
   else:ev['score_hold']='automatic_Y_boundary_unconfirmed'
   events.append(ev)
 tracker.finish();out.append(dict(candidate=name,events=events,final=tracker.status()))
assert sha(p)==original
(O/'saved-session.json').write_text(json.dumps(dict(scope='unlabelled_no_video_observational_diagnostic_not_ground_truth',source_sha256=original,state_bundle_sha256=sha(bp),state_bundle_activated_in_live=False,candidate_code_sha256=sha(B/'tools/ytw-work/ytw_hold_onset_v4.py'),results=out),ensure_ascii=False,indent=2)+'\n')
print([(r['candidate'],len(r['events']),[(e['start_tick'],e['end_tick'],e.get('Y_boundary_evidence')) for e in r['events']]) for r in out])
