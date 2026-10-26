"""읽기 전용 원본 재생. 후보 출력만 별도 경로에 저장한다."""
import sys,json,hashlib
from pathlib import Path
from fractions import Fraction
from collections import Counter
import numpy as np,joblib
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/ytw-work-20261024'
APP=B/'output/live-webcam-20261023';SNAP=APP/'runtime/c76334937f21ef24a27aa1c1687de6c9577bf6249e5cac606151a328893746f7'
sys.path[:0]=[str(APP),str(SNAP)]
from ytw_candidate import LearnedPhaseSource,YTWTracker,score_event,REQUIRED
from app.scoring.engine import ScoringEngine
from live_observation import mask_frame_border
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
bundle_path=B/'output/initial-pose-return-v7-20261024/initial-pose-return-bundle.joblib'
bundle=joblib.load(bundle_path);source=LearnedPhaseSource(bundle);engine=ScoringEngine();out=[]
inventory={r['id']:r for r in json.load(open(B/'output/r2d2-review-20261024/inventory.json'))}
policy=json.load(open(B/'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json'))
inputs=json.load(open(B/'output/initial-pose-return-v7-20261024/state-inputs.frozen.json'))
paths=[('saved_live_497',APP/'sessions/1792846426614521826/pose.npz')]+[(sid,B/f'output/integrated-five-20261024/team-observations/{sid}/side.npz') for sid in ['V37','V39','V41','V43']]
for sid,path in paths:
 z=np.load(path);xy=z['image_xy'];digest=sha(path)
 if sid=='saved_live_497':
  width,height=int(z['width']),int(z['height']);ticks=z['frame_numbers'];role='unlabelled_live_diagnostic_not_ground_truth'
 else:
  r=inventory[sid];rot=abs(float(r.get('rotation',[{}])[0].get('rotation',0)))%180
  width,height=(r['height'],r['width']) if rot==90 else (r['width'],r['height'])
  ticks=np.rint(z['frame_numbers']/float(Fraction(r['fps']))*10).astype(int)
  role=policy['assignments'][sid]+'_already_used_state_training_replay'
  rows=[r for r in inputs if r['source_id']==sid]
  assert all(r['cache_sha256']==digest for r in rows)
  assert json.load(open(path.with_suffix('.json')))['config']['source_sha256']==sha(r['copy'])
 tracker=YTWTracker(source);events=[];labels=Counter();ready=0;valid=0
 for tick,points in zip(ticks,xy):
  p=mask_frame_border(points,width,height)
  if np.isfinite(p[REQUIRED]).all():
   label,is_ready=source(p);labels[label]+=1;ready+=int(is_ready);valid+=1
  e=tracker.push(int(tick),points,width,height)
  if e:
   scores=score_event(engine,e)
   events.append(dict(**{k:v for k,v in e.items() if k!='samples'},score=scores))
 before_finish=tracker.status();tracker.finish()
 assert sha(path)==digest
 out.append(dict(id=sid,input=str(path),input_sha256=digest,role=role,samples=len(xy),
                 width=width,height=height,border_safe_required_samples=valid,phase_decisions=dict(labels),ready_decisions=ready,
                 events=events,before_finish=before_finish,final=tracker.status(),raw_cache_view_name_not_protocol_authority=sid!='saved_live_497'))
result=dict(schema='ytw-candidate-replay-v1',state_bundle=str(bundle_path),state_bundle_sha256=sha(bundle_path),
            scoring_snapshot=str(SNAP),state_gate_is_development_only=True,models_refit=False,
            labels_or_splits_modified=False,production_replacement=False,independent_final_test=False,results=out)
(O/'replay.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps([dict(id=x['id'],samples=x['samples'],valid=x['border_safe_required_samples'],phase=x['phase_decisions'],count=x['final']['count'],rejects=x['final']['rejected']) for x in out],ensure_ascii=False,indent=2))
