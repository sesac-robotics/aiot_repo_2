"""Replay frozen frame predictions on immutable original pose caches.

No fitting/camera/live mutations. Verify v5 semantics equal v4.1 segmentation.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib
from pathlib import Path
from fractions import Fraction
from collections import Counter
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/ytw-work-20261024/readiness-v5-20261025';V=B/'output/ytw-work-20261024/y-entry-v4.1';A=B/'output/live-webcam-20261023';sys.path.insert(0,str(A))
from ytw_hold_onset_v4 import YTWTracker as Parent
from ytw_readiness_v5 import YTWTracker as Candidate,scoring_input_hold
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.load(open(O/'plan.frozen.json'));assert sha(B/'tools/ytw-work/ytw_hold_onset_v4.py')==plan['parent_boundary_code_sha256']
trace_rows=json.load(open(V/'phase-traces.json'));I={r['id']:r for r in json.load(open(B/'output/r2d2-review-20261024/inventory.json'))};policy=json.load(open(B/'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json'));models=json.load(open(B/'output/ytw-work-20261024/logo-v1/replay.json'))
all_reports=[]
def core(event):return {k:event[k] for k in ['start_tick','end_tick','auto_phase_windows','observation_coverage','temporal_slot_coverage','Y_boundary_evidence']}
for sid in plan['source_ids']:
 receipt=next(r for r in trace_rows if r['source_id']==sid);cp=B/f'output/integrated-five-20261024/team-observations/{sid}/side.npz';s=I[sid];assert sha(cp)==receipt['cache_sha256'];assert sha(s['copy'])==receipt['source_sha256'];assert policy['assignments'][sid]=='train'
 f=next(f for f in models['folds'] if f['excluded_person']==receipt['person']);assert sha(f['model'])==receipt['fold_model_sha256'];assert receipt['person'] not in f['train_people']
 z=np.load(cp);fps=float(Fraction(s['fps']));ticks=np.rint(z['frame_numbers']/fps*10).astype(int);rot=abs(float(s['rotation'][0]['rotation']))%180;w,h=(s['height'],s['width']) if rot==90 else (s['width'],s['height']);pred={r['tick']:(r['phase'],r['ready']) for r in receipt['trace']};assert set(ticks)==set(pred)
 def replay(cls,cutoff=None,keep_status=False):
  stamp=[0];tracker=cls(lambda p:pred[int(stamp[0])]);events=[];states=[]
  for tick,points in zip(ticks,z['image_xy']):
   if cutoff is not None and tick>cutoff:break
   stamp[0]=tick;e=tracker.push(int(tick),points,w,h)
   if e:events.append(e)
   if keep_status:states.append(dict(tick=int(tick),state=tracker.status()))
  return tracker,events,states
 before,old,_=replay(Parent);after,new,statuses=replay(Candidate,keep_status=True)
 assert before.count==after.count and len(old)==len(new)
 for a,b in zip(old,new):
  assert core(a)==core(b)
  assert [t for t,p in a['samples']]==[t for t,p in b['samples']]
  assert np.allclose([p for t,p in a['samples']],[p for t,p in b['samples']],equal_nan=True)
  assert scoring_input_hold(b)['input_contract_eligible']
  r=b['readiness_receipts'];assert r['Y_phase_candidate_at_tick']<=r['Y_ready_confirmed_at_tick']<=r['ordered_sequence_confirmed_at_tick'];assert r['Y_stable_confirmed_at_tick']<=r['ordered_sequence_confirmed_at_tick'];assert r['ordered_sequence_confirmed_at_tick']==b['end_tick']
 # Every event must be absent one tick before completion and present at that tick.
 for index,e in enumerate(new):
  t=e['end_tick'];_,prev,_=replay(Candidate,t-1);_,now,_=replay(Candidate,t)
  assert len(prev)==index and len(now)==index+1 and core(now[-1])==core(e)
  assert now[-1]['readiness_receipts']==e['readiness_receipts']
 for x in statuses:
  state=x['state']
  if not state['current_observation']['valid']:assert not state['current_Y_ready']
  if state['current_Y_ready']:assert state['current_Y_ready_streak']>=5 and state['current_observation']['phase_label']=='Y'
 note=dict(source_id=sid,source_role='train_person_excluded_exposed_development',source_sha256=receipt['source_sha256'],cache_sha256=receipt['cache_sha256'],fold_model_sha256=receipt['fold_model_sha256'],attempt_count=len(new),segmentation_and_dense_score_input_identical_to_v4_1=True,prefix_causality_verified_for_all_events=True,current_state_counts=dict(Counter(x['state']['current_state'] for x in statuses)),unobserved_frames=sum(not x['state']['current_observation']['valid'] for x in statuses),ready_reported_on_unobserved_frames=0,events=[{k:v for k,v in e.items() if k!='samples'} for e in new],status_timeline=statuses)
 all_reports.append(note);print(sid,len(new),'events; parity/prefix passed',flush=True)
result=dict(schema='ytw-current-readiness-receipts-v5-development',plan_sha256=sha(O/'plan.frozen.json'),scope=plan['exposure'],all_sources_unchanged=True,parameters_or_model_changed=False,production_enabled=False,Test_used=False,source_reports=all_reports,summary=dict(sources=len(all_reports),ordered_attempts=sum(x['attempt_count'] for x in all_reports),segmentation_changed=False,prefix_event_checks=sum(x['attempt_count'] for x in all_reports),current_state_counts=dict(sum((Counter(r['current_state_counts']) for r in all_reports),Counter())),unobserved_frames=sum(x['unobserved_frames'] for x in all_reports),ready_on_unobserved_frames=0),candidate_code_sha256=sha(B/'tools/ytw-work/ytw_readiness_v5.py'))
(O/'replay.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result['summary'],ensure_ascii=False,indent=2))
