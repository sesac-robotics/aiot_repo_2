"""Select for genuine mild Train errors using participant-separated OOF only."""
import collections,json,sys,datetime,copy
from pathlib import Path
import numpy as np,joblib
from sklearn.model_selection import GroupKFold
B=Path(__file__).resolve().parents[2];O=B/'output/mild-return-priority-20261024';BASE=B/'output/mild-return-fit-20261024';O.mkdir(exist_ok=True);sys.path.insert(0,str(B/'tools/error-enrichment'))
from train_scores import fit,predictions,metrics,weights
from acceptance import assess
from common import sha
read=lambda p:json.loads(p.read_text());rows=read(BASE/'score-inputs.frozen.json');subset=read(BASE/'changed-heads-fit/score-inputs.frozen.json');cues=[]
for r in rows:
 if r['split']!='train':continue
 for n,g in zip(r['items'],r['levels']):
  if n['bank']=='height_upper_reversal_v1' and n['eligible'] and g is not None and g<2:cues.append((max(n['feature'][0],n['feature'][3]),r['id']))
mild=[i for v,i in sorted(cues)[:(len(cues)+1)//2]];assert len(mild)==5
for n in ['score-inputs.frozen.json','test-inputs.frozen.json','label-revisions.frozen.json']:
 assert not (O/n).exists();(O/n).write_bytes((BASE/n).read_bytes())
(O/'MODEL_SELECTION_PROTOCOL.frozen.json').write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_run=str(B/'output/return-root-cause-v3-20261024'),data_parent=str(BASE),mild_Train_ids=mild,mild_definition='lower half of observed error amplitudes max(hip,shoulder) from Train upper-reversal geometry; not development or Test cutoffs',normal_false_alarm_cap=.15,selection='Train participant-group OOF mild error recall first, normal FP cap and overall error recall; fixed development only assessed after selection',bank_selection_policy='Train_mild_priority',source_roles_unchanged=True,Test_input_sha256=sha(O/'test-inputs.frozen.json'),training_input_sha256=sha(O/'score-inputs.frozen.json'),code_sha256=sha(Path(__file__)),production_replacement=False),indent=2))
buckets=collections.defaultdict(list)
for r in subset:
 for j,(n,g) in enumerate(zip(r['items'],r['levels'])):
  if g is not None and n['eligible'] and n['feature'] is not None:buckets[(r['exercise'],n['item'],n['bank'])].append(dict(id=r['id'],exercise=r['exercise'],index=j,x=n['feature'],target=g,person=r['person'],group=r['group'],family=r['family'],origin=r['origin'],split=r['split'],sources=r['sources']))
reports=[];heads={}
for key,rs in sorted(buckets.items()):
 tr=[r for r in rs if r['split']=='train'];va=[r for r in rs if r['split']=='validation'];groups=np.array([r['group'] for r in tr]);assert not set(groups)&{r['group'] for r in va};folds=list(GroupKFold(n_splits=min(4,len(set(groups)))).split(tr,groups=groups));candidates=[]
 for name in ['tree2','tree4','tree8','boost80','ridge1','ridge10','ordinal0.1','ordinal1','ordinal10','grade0.1','grade1','grade10']:
  ps=[];base=[]
  for ti,vi in folds:
   a=[tr[i] for i in ti];b=[tr[i] for i in vi];m=fit(a,name);ps+=predictions(m,b);mean=float(np.average([r['target'] for r in a],weights=weights(a)));base += [dict(id=r['id'],group=r['group'],target=r['target'],prediction=mean) for r in b]
  met=metrics(ps);mp=[p for p in ps if p['id'] in mild];mm=metrics(mp);coverage=len(mp);recall=mm['error_recall'] if mm else 0.;fp=met['normal_false_alarm'];valid=coverage==len(mild) and met['error_recall']>=.65 and fp<=.15
  rank=[0 if valid else 1,1-recall+(len(mild)-coverage)/len(mild),fp,met['balanced_group_MAE']]
  candidates.append(dict(name=name,metrics=met,mild_metrics=mm,mild_coverage=coverage,priority_rank=rank,predictions=ps,constant_baseline=metrics(base)))
 best=min(candidates,key=lambda c:c['priority_rank']);model=fit(tr,best['name']);vp=predictions(model,va);gate=assess(best['predictions'],vp,best['constant_baseline'],None);counts=collections.Counter(r['target'] for r in tr)
 heads[key]=dict(model=model,key=key,feature_dimensions=len(tr[0]['x']),grade_support=dict(counts),trained_ids=[r['id'] for r in tr],candidate_gate_passed=gate['passed'],production_replacement=False)
 reports.append(dict(key=list(key),train_count=len(tr),validation_count=len(va),grade_counts=dict(counts),normal_groups=len({r['group'] for r in tr if r['target']==2}),error_groups=len({r['group'] for r in tr if r['target']<2}),source_families=dict(collections.Counter(r['family'] for r in tr)),status='trained_experimental',selected=best['name'],group_cv=best['metrics'],constant_group_cv=best['constant_baseline'],mild_group_cv=best['mild_metrics'],mild_coverage=best['mild_coverage'],train_mild_priority_rank=best['priority_rank'],mild_Train_ids=mild,fixed_development_validation=metrics(vp),candidate_gate_passed=gate['passed'],runtime_replacement=False,acceptance=gate,selected_oof_predictions=best['predictions'],validation_predictions=vp,candidate_cv=[{k:v for k,v in c.items() if k!='predictions'} for c in candidates]))
 print('SELECTED',key,best['name'],'mild',best['mild_metrics'],'CV',best['metrics'],'DEV',metrics(vp),flush=True)
SUB=O/'changed-heads-fit';SUB.mkdir(exist_ok=True);joblib.dump(dict(schema='integrated-five-criterion-candidates-v1',heads=heads,labels_sha256=sha(O/'score-inputs.frozen.json'),runtime_replacement=False),SUB/'score-candidates.joblib');(SUB/'score-training.json').write_text(json.dumps(dict(heads=reports,completed=True,trained_heads=len(heads),gate_passed=sum(h['candidate_gate_passed'] for h in reports),final_test_used=False),indent=2));print('COMPLETE',len(heads),flush=True)
