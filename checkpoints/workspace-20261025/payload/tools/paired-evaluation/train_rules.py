"""Freeze an explicit Train-calibrated Rule comparator alongside the frozen AI."""
import sys,json,collections,datetime,os
from pathlib import Path
import numpy as np,joblib
from sklearn.model_selection import GroupKFold
B=Path('/home/jayden/SeSACPAI-Mid');BASE=Path(os.environ.get('PAIRED_AI_RUN',str(B/'output/pushup-repaired-five-v2-20261024')));O=Path(os.environ.get('PAIRED_RULE_OUTPUT',str(B/'output/ai-rule-paired-v2-20261024')));O.mkdir(exist_ok=True);sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from train_scores import metrics
from selection import choose
from thresholds import allowed_indices,fit,predict,transform_input
rows=json.loads((BASE/'score-inputs.frozen.json').read_text());ai=joblib.load(BASE/'score-candidates.joblib');buckets=collections.defaultdict(list)
for r in rows:
 for i,(item,grade) in enumerate(zip(r['items'],r['levels'])):
  key=(r['exercise'],item['item'],item['bank'])
  if key not in ai['heads'] or not item['eligible'] or grade is None or item['feature'] is None:continue
  buckets[key].append(dict(id=r['id'],person=r['person'],group=r['group'],family=r['family'],exercise=r['exercise'],index=i,origin=r['origin'],split=r['split'],sources=r['sources'],target=grade,x=transform_input(item['bank'],item['item'],item['feature'])[0].tolist(),raw_dimensions=len(item['feature'])))
heads=[];reports=[]
for key,rs in sorted(buckets.items()):
 train=[r for r in rs if r['split']=='train'];validation=[r for r in rs if r['split']=='validation'];indices=allowed_indices(key[2],key[1],rs[0]['raw_dimensions'])
 if not indices:reports.append(dict(key=key,status='unsupported_no_interpretable_scalar'));continue
 assert not {r['group'] for r in train}&{r['group'] for r in validation}
 groups=np.array([r['group'] for r in train]);splits=list(GroupKFold(min(4,len(set(groups)))).split(train,groups=groups));candidates=[]
 for idx in indices:
  for direction in [-1,1]:
   ps=[]
   for ti,vi in splits:
    t=[train[i] for i in ti];v=[train[i] for i in vi];rule=fit(t,idx,direction);values=predict(rule,[r['x'] for r in v]);ps.extend(dict(id=r['id'],person=r['person'],group=r['group'],target=r['target'],prediction=float(p)) for r,p in zip(v,values))
   candidates.append(dict(index=idx,direction=direction,metrics=metrics(ps),predictions=ps))
 selected=choose(candidates);rule=fit(train,selected['index'],selected['direction']);vp=predict(rule,[r['x'] for r in validation]) if validation else [];vps=[dict(id=r['id'],group=r['group'],target=r['target'],prediction=float(p)) for r,p in zip(validation,vp)];head=dict(key=key,rule=rule,feature_dimensions=rs[0]['raw_dimensions'],geometry_transform_version='criterion_aligned_measurements_v2',training_ids=[r['id'] for r in train],uses_ai_output=False);heads.append(head);reports.append(dict(key=key,status='trained_explicit_thresholds',selected_rule=rule,training_group_cv=selected['metrics'],fixed_development=metrics(vps),selected_oof_predictions=selected['predictions'],validation_predictions=vps,candidates=[{k:v for k,v in c.items() if k!='predictions'} for c in candidates]));print('RULE_FROZEN',key,len(train),flush=True)
value=dict(schema='target_aligned_train_calibrated_scalar_rules_v2',geometry_transform_version='criterion_aligned_measurements_v2',implementation_sha256=sha(B/'tools/paired-evaluation/thresholds.py'),heads=heads,source_ai_model_sha256=sha(BASE/'score-candidates.joblib'),training_inputs_sha256=sha(BASE/'score-inputs.frozen.json'),policy_sha256=sha(B/'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json'),frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),not_existing_deployed_rules=True,test_used_for_selection=False,production_replacement=False)
p=O/'rules.frozen.json';assert not p.exists(),'Never overwrite a frozen comparator';p.write_text(json.dumps(value,indent=2));(O/'rule-training.json').write_text(json.dumps(dict(completed=True,heads=reports,test_used=False,legacy_rules_unchanged=True),indent=2));print('RULE_TRAINING_COMPLETE',len(heads),flush=True)
