"""Freeze a bounded Train-only, diverse audit batch before any new fit."""
from pathlib import Path
import json,sys,hashlib,collections,cv2,numpy as np
sys.dont_write_bytecode=True
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/four-exercise-review-train-v1-20261025';R=B/'output/squat-fixed-reference-combined-v3-20261024';EX=['shoulder','pushups','ytw','plank'];sys.path[:0]=[str(B/'tools/error-enrichment')];from cache_sequences import Resolver;from split_guard import assert_training_allowed
rows=json.loads((R/'score-inputs.frozen.json').read_text());report=json.loads((R/'score-training.json').read_text());sources=[json.loads(l) for l in (B/'project-documents/project-handbook/data-audit-20261025/catalogs/held-media.jsonl').open()];sha_paths=collections.defaultdict(list)
for r in sources:
 if r.get('sha256') and r['exists']:sha_paths[r['sha256']].append(r['path'])
linked={r['id']:r for ex in EX for r in [json.loads(l) for l in (B/f'project-documents/project-handbook/data-audit-20261025/{ex}/current-labelled-inputs.jsonl').open()]};res=Resolver();priority=collections.defaultdict(float)
for h in report['heads']:
 for p in h.get('selected_oof_predictions',[]):
  if (p['target']==2 and p['prediction']<1.5) or (p['target']<2 and p['prediction']>=1.5):priority[p['id']]=max(priority[p['id']],abs(p['target']-p['prediction']))
selected=[];rejected=[];candidate_cache={}
for ex in EX:
 pool=[r for r in rows if r['exercise']==ex and r['split']=='train'];assert_training_allowed(pool);people=collections.Counter();families=collections.Counter();chosen=[];seen_sources=collections.Counter()
 # Error and normal examples are both reviewed. OOF is a queue signal only.
 pool.sort(key=lambda r:(-priority[r['id']],sum(g is not None and g<2 for g in r['levels'])==0,r['person'],r['id']))
 for passno in range(3):
  for r in pool:
   if len(chosen)>=16:break
   if any(v['id']==r['id'] for v in chosen):continue
   if people[r['person']]>= (2 if passno<2 else 3):continue
   if passno==0 and families[r['family']]>=6:continue
   if any(seen_sources[h]>=2 for h in r['sources']):continue
   try:
    pairs=res.resolve(r)
    if not pairs:raise ValueError('no_source_verified_cache')
    seq,p=pairs[0];m=json.loads(p.with_suffix('.json').read_text());z=np.load(p);cfg=m['config'];sha=cfg['source_sha256'];assert sha in r['sources'];assert str(z['fingerprint'])==m['fingerprint'];assert hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()==m['fingerprint'];paths=sha_paths[sha]+linked[r['id']]['source_paths'];paths=[q for q in dict.fromkeys(paths) if Path(q).exists() and Path(q).suffix.lower() in ['.mp4','.mkv','.mov','.avi','.webm']];assert paths
    src=next((q for q in paths if '/Dataset Exercise-wise/' in q),paths[0]);cap=cv2.VideoCapture(src);fps=cap.get(5);frame_count=int(cap.get(7));ok,im=cap.read();cap.release();assert ok
    fs=z['frame_numbers'];times=fs/float(cfg['segment']['fps']);start=float(times[0]);end=float(times[-1]);mask=np.ones(len(fs),bool)
    if r['family']=='R2D2':
     lab=res.team[r['id'][5:]];start=lab['start'];end=lab['end'];mask=(times>=start)&(times<end)
    if r['id'] in res.new:
     lab=res.new[r['id']];start=lab['start'];end=lab['end'];mask=(times>=start)&(times<end)
    ids=np.flatnonzero(mask);assert len(ids)>=8;assert end-start<=12,'window_too_long_for_rep_audit';assert abs(float(cfg['segment']['fps'])-fps)<.05
    d=dict(id=r['id'],exercise=ex,family=r['family'],person=r['person'],group=r['group'],split='train',sources=r['sources'],source_sha256=sha,source_path=src,fps=fps,width=im.shape[1],height=im.shape[0],window_seconds=[start,end],cache_path=str(p),cache_fingerprint=m['fingerprint'],cache_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),actual_native_frames=fs[mask].tolist(),cache_indices=ids.tolist(),previous_levels=r['levels'],previous_items=[x['item'] for x in r['items']],previous_banks=[x['bank'] for x in r['items']],selection='Train_OOF_error_or_diverse_control_not_label_source',rendered_blind_to_model_prediction=True)
    chosen.append(d);people[r['person']]+=1;families[r['family']]+=1
    for h in r['sources']:seen_sources[h]+=1
   except Exception as e:rejected.append(dict(id=r['id'],reason=str(e)))
  if len(chosen)>=16:break
 assert len(chosen)==16,(ex,len(chosen));selected+=chosen;print(ex,dict(families),len(people),'people',flush=True)
assert_training_allowed(selected)
protocol=dict(schema='four-exercise-Train-review-v1',exercises=EX,selection_limit_per_exercise=16,source_of_labels='new_direct_source_frame_review_not_model_predictions',label_target='existing_project_item_criteria_with_variant_separation',grade_values={'2':'observed_criterion_met','1':'observed_partial_deficit','0':'observed_clear_deficit','null':'unobservable_or_protocol_mismatch'},evaluation=dict(type='Train_only_person_source_connected_GroupKFold_fixed_4fold',fixed_models=['balanced_logistic_C1','ExtraTrees64_minleaf3'],selection='no_new_Test_reading; compare_new_labels_and_parent_models_on_held_groups',adoption='No live promotion from Train review CV alone; record error_recall/normal_false_alarm/abstentions and group support',score_training_only_Train=True),squat_changed=False)
for name,val in [('protocol.frozen.json',protocol),('review-plan.frozen.json',selected),('selection-exclusions.json',rejected)]:
 p=O/name;assert not p.exists();p.write_text(json.dumps(val,ensure_ascii=False,indent=2)+'\n')
