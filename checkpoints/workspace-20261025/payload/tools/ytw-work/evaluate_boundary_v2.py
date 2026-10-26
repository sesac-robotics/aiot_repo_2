"""Evaluate fixed existing fold models on video references without tuning."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib
from pathlib import Path
from fractions import Fraction
from collections import Counter
import numpy as np,joblib
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/ytw-work-20261024/boundary-v2';L=B/'output/ytw-work-20261024/logo-v1';A=B/'output/live-webcam-20261023';S=A/'runtime/c76334937f21ef24a27aa1c1687de6c9577bf6249e5cac606151a328893746f7'
sys.path[:0]=[str(A),str(S)]
from ytw_candidate import YTWTracker
from app.scoring.initial_pose_return import pose
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ref=json.load(open(O/'references-v2.frozen.json'));refhash=sha(O/'references-v2.frozen.json');plan=json.load(open(O/'plan.frozen.json'));assert sha(A/'ytw_candidate.py')==plan['tracker_frozen_sha256']
prior=json.load(open(L/'replay.json'));models={f['excluded_person']:f for f in prior['folds']};I={r['id']:r for r in json.load(open(B/'output/r2d2-review-20261024/inventory.json'))};policy=json.load(open(B/'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json'));labels=json.load(open(B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json'))['labels']
all_events={f['source_id']:f['events'] for f in prior['folds']};new=[]
for sid in ['V38','V40','V44']:
 r=I[sid];assert policy['assignments'][sid]=='train';assert sha(r['copy'])==r['sha256']==ref['source_video_sha256'][sid]
 person='r2d2_'+next(x['person'] for x in labels if x['source_id']==sid);f=models[person];assert person not in f['train_people'];assert sha(f['model'])==f['model_sha256'];bundle=joblib.load(f['model']);heads=bundle['heads']
 cp=B/f'output/integrated-five-20261024/team-observations/{sid}/side.npz';ch=sha(cp);meta=json.load(open(cp.with_suffix('.json')));assert meta['config']['source_sha256']==r['sha256'];z=np.load(cp);fps=float(Fraction(r['fps']));ticks=np.rint(z['frame_numbers']/fps*10).astype(int)
 rot=abs(float(r.get('rotation',[{}])[0].get('rotation',0)))%180;width,height=(r['height'],r['width']) if rot==90 else (r['width'],r['height'])
 predictions=[]
 def source(xy):
  vector=pose('ytw',xy)
  if vector is None:return None,False
  return str(heads['phase'].predict([vector])[0]),bool(heads['target'].predict([vector])[0]==1)
 tracker=YTWTracker(source);events=[]
 for tick,xy in zip(ticks,z['image_xy']):
  e=tracker.push(int(tick),xy,width,height)
  if e:events.append({k:v for k,v in e.items() if k!='samples'})
 tracker.finish();all_events[sid]=events
 new.append(dict(source_id=sid,person=person,excluded_model=f['model'],model_sha256=f['model_sha256'],source_sha256=r['sha256'],cache_sha256=ch,source_role='train_unchanged_excluded_person_development',events=events,final=tracker.status()))
 assert sha(cp)==ch
 print(sid,'candidate_events',len(events),flush=True)
# Association permits late/early boundaries for measurement; never adjusts labels.
positives=[]
def error(pred,interval):
 lo,hi=interval;return dict(predicted_seconds=pred,reference_bracket=interval,min_error_seconds=round(max(lo-pred,pred-hi,0),4),max_error_seconds=round(max(abs(pred-lo),abs(pred-hi)),4),definitely_within_two_slots=max(abs(pred-lo),abs(pred-hi))<=.200001,possibly_within_two_slots=max(lo-pred,pred-hi,0)<=.200001)
for p in ref['positive_attempts']:
 w=p['phase_entry_seconds'];candidates=[e for e in all_events[p['source_id']] if w['Y'][0]-.8<=e['start_tick']/10<=w['T'][0] and w['T'][0]<=e['end_tick']/10<=w['W'][1]+1.5]
 row=dict(reference_id=p['id'],source_id=p['source_id'],candidate_count=len(candidates),matched=len(candidates)==1)
 if len(candidates)==1:
  e=candidates[0];row['event_ticks']=[e['start_tick'],e['end_tick']];row['phase_errors']={name:error((e['start_tick']+e['auto_phase_windows'][name][0])/10,w[name]) for name in 'YTW'};row['terminal_error']=error(e['end_tick']/10,p['confirmation_seconds'])
 positives.append(row)
negative=[]
for n in ref['natural_negative_episodes']:
 a,b=n['interval_seconds'];events=[e for e in all_events[n['source_id']] if a<=e['end_tick']/10<=b]
 negative.append(dict(reference_id=n['id'],source_id=n['source_id'],kind=n['kind'],interval_seconds=[a,b],false_completion_events=[[e['start_tick'],e['end_tick']] for e in events],false_completion=bool(events)))
phase=[v for p in positives if p['matched'] for v in p['phase_errors'].values()];terminal=[p['terminal_error'] for p in positives if p['matched']]
summary=dict(ordered_attempt_drafts=10,matched_draft_attempts=sum(p['matched'] for p in positives),phase_boundary_comparisons=len(phase),phase_boundaries_definitely_within_0_2s=sum(x['definitely_within_two_slots'] for x in phase),phase_boundaries_possibly_within_0_2s=sum(x['possibly_within_two_slots'] for x in phase),terminal_comparisons=len(terminal),terminals_definitely_within_0_2s=sum(x['definitely_within_two_slots'] for x in terminal),terminals_possibly_within_0_2s=sum(x['possibly_within_two_slots'] for x in terminal),natural_negative_episodes=len(negative),negative_episodes_with_false_completion=sum(x['false_completion'] for x in negative),negative_kind_counts=dict(Counter(x['kind'] for x in negative)))
result=dict(schema='ytw-fixed-fold-video-reference-development-v2',reference_sha256=refhash,scope='exposed_AI_video_draft_person_excluded_fixed_settings_development_not_final_accuracy',parameters_changed=False,labels_used_for_fit=False,prior_fold_models_reused=True,positive_comparisons=positives,negative_comparisons=negative,new_source_replays=new,summary=summary,activation_gate_passed=False,holds=['negative_episode_count13_below20','phase_reference_uncertainty_and_AI_draft_not_expert_ground_truth','ordered_completion_draft_not_normal_quality_truth','existing_v7_hyperparameters_previously_chosen_with_all_people','no_new_unseen_person_for_full_bundle','model_score_quality_not_evaluated'],models_deployed=False,Test_opened=False)
assert sha(O/'references-v2.frozen.json')==refhash
(O/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
