"""Fixed four-person-out YTW development. No held-out Test reads or live edits."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib
from pathlib import Path
from fractions import Fraction
from collections import Counter
from datetime import datetime,timezone
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesClassifier
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/ytw-work-20261024/logo-v1';APP=B/'output/live-webcam-20261023'
S=APP/'runtime/c76334937f21ef24a27aa1c1687de6c9577bf6249e5cac606151a328893746f7'
sys.path[:0]=[str(APP),str(S),str(B/'tools/error-enrichment')]
from split_guard import assert_training_allowed,POLICY
from ytw_candidate import YTWTracker,REQUIRED
from app.scoring.initial_pose_return import pose
from live_observation import mask_frame_border
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
P=B/'output/initial-pose-return-v7-20261024/state-inputs.frozen.json';allrows=json.load(open(P));rows=[r for r in allrows if r['exercise']=='ytw'];assert_training_allowed(rows)
inv={r['id']:r for r in json.load(open(B/'output/r2d2-review-20261024/inventory.json'))}
plan=dict(schema='ytw-logo-development-plan-v1',frozen_at_utc=datetime.now(timezone.utc).isoformat(),inputs=str(P),inputs_sha256=sha(P),policy_sha256=sha(POLICY),rows=76,people=sorted({r['person'] for r in rows}),heads=['target','phase'],algorithm=dict(name='ExtraTreesClassifier',n_estimators=128,min_samples_leaf=2,max_features=1.,random_state=241024,n_jobs=1),weighting='inverse_class_person_count_as_original_train.py',tracker=dict(dwell_slots=5,missing_reset_slots=4,max_gap_slots=3,attempt_max_slots=300,required_coverage=.8,border_fraction=.02),gates=dict(min_unseen_people=2,min_completed_attempts=10,min_negative_intervals=20,min_detection_rate=.9,max_false_complete_rate=.05,min_boundaries_within_two_slots=.9),ground_truth='video_review_required_before_accuracy_metrics',setting_selection_exposure='v7_settings_previously_selected_with_all_four_people_not_nested_independent' ,final_test_opened=False,production_replacement=False)
planpath=O/'plan.frozen.json'
if planpath.exists():
 old=json.load(open(planpath));assert old['inputs_sha256']==plan['inputs_sha256'];plan=old
else:planpath.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
reports=[]
for person in plan['people']:
 tr=[r for r in rows if r['person']!=person];va=[r for r in rows if r['person']==person]
 assert set(r['source_sha256'] for r in tr).isdisjoint(r['source_sha256'] for r in va);assert_training_allowed(tr)
 heads={}
 for target in ['target','phase']:
  rs=[r for r in tr if r[target] is not None];counts=Counter((r[target],r['person']) for r in rs);ys=Counter(y for y,p in counts)
  if set(r[target] for r in rs) != ({0,1} if target=='target' else {'Y','T','W'}):
   raise ValueError('Missing fold classes: unsupported, no constant model allowed')
  weights=[1/(ys[r[target]]*counts[r[target],r['person']]) for r in rs]
  model=ExtraTreesClassifier(**{k:v for k,v in plan['algorithm'].items() if k!='name'})
  model.fit([r['x'] for r in rs],[r[target] for r in rs],sample_weight=weights);heads[target]=model
 sid=va[0]['source_id'];cp=Path(va[0]['cache']);cachehash=sha(cp);assert all(r['cache_sha256']==cachehash for r in va)
 r=inv[sid];assert sha(r['copy'])==va[0]['source_sha256'];z=np.load(cp)
 fps=float(Fraction(r['fps']));ticks=np.rint(z['frame_numbers']/fps*10).astype(int)
 rot=abs(float(r.get('rotation',[{}])[0].get('rotation',0)))%180;width,height=(r['height'],r['width']) if rot==90 else (r['width'],r['height'])
 predictions=[]
 def source(p):
  x=pose('ytw',p)
  if x is None:return None,False
  return str(heads['phase'].predict([x])[0]),bool(heads['target'].predict([x])[0]==1)
 tracker=YTWTracker(source);events=[]
 for tick,xy in zip(ticks,z['image_xy']):
  p=mask_frame_border(xy,width,height);phase,ready=source(p)
  predictions.append(dict(tick=int(tick),phase=phase,ready=ready,observed=bool(np.isfinite(p[REQUIRED]).all())))
  event=tracker.push(int(tick),xy,width,height)
  if event:events.append({k:v for k,v in event.items() if k!='samples'})
 tracker.finish();mp=O/(person+'-models.joblib');joblib.dump(dict(heads=heads,train_ids=[r['id'] for r in tr],excluded_person=person,production_replacement=False),mp)
 labelled=[]
 for a in va:
  labelled.append(dict(id=a['id'],phase_truth=a['phase'],phase_prediction=str(heads['phase'].predict([a['x']])[0]),ready_truth=a['target'],ready_prediction=int(heads['target'].predict([a['x']])[0])))
 reports.append(dict(excluded_person=person,source_id=sid,source_sha256=va[0]['source_sha256'],cache=str(cp),cache_sha256=cachehash,model=str(mp),model_sha256=sha(mp),train_people=sorted({r['person'] for r in tr}),train_ids=[r['id'] for r in tr],excluded_ids=[r['id'] for r in va],frame_predictions=predictions,labelled_frame_predictions=labelled,events=events,final=tracker.status(),sequence_ground_truth_reviewed=False,accuracy_metrics=None))
 print(person,sid,'candidate_events',len(events),flush=True)
 assert sha(cp)==cachehash
result=dict(plan_sha256=sha(planpath),scope='leave_one_person_out_development_not_final_Test',production_replacement=False,folds=reports,sequence_validation_gate_passed=False,gate_reason='continuous_video_truth_and_negative_denominators_not_yet_reviewed')
(O/'replay.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
