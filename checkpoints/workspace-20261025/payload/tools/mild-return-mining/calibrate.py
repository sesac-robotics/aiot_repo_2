"""Nested Train-group calibration; no development/Test-driven cutoffs."""
import collections,copy,json,sys,datetime
from pathlib import Path
import numpy as np,joblib
from sklearn.model_selection import GroupKFold
B=Path(__file__).resolve().parents[2];BASE=B/'output/mild-return-fit-20261024';O=B/'output/mild-return-calibrated-20261024';O.mkdir(exist_ok=True);SUB=O/'changed-heads-fit';SUB.mkdir(exist_ok=True);sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from train_scores import fit,metrics,weights,predictions
from acceptance import assess
from models import NormalThresholdRegressor
read=lambda p:json.loads(p.read_text());rows=read(BASE/'score-inputs.frozen.json');subset=read(BASE/'changed-heads-fit/score-inputs.frozen.json');cues=[]
for r in rows:
 if r['split']=='train':
  for n,g in zip(r['items'],r['levels']):
   if n['bank']=='height_upper_reversal_v1' and n['eligible'] and g is not None and g<2:cues.append((max(n['feature'][0],n['feature'][3]),r['id']))
mild={i for a,i in sorted(cues)[:(len(cues)+1)//2]};assert len(mild)==5
for n in ['score-inputs.frozen.json','test-inputs.frozen.json','label-revisions.frozen.json']:
 assert not (O/n).exists();(O/n).write_bytes((BASE/n).read_bytes())
protocol=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_run=str(B/'output/return-root-cause-v3-20261024'),data_parent=str(BASE),mild_Train_ids=sorted(mild),mild_definition='lower half of Train error max(hip,shoulder) amplitude; not a development cutoff',selection='nested participant-group CV on Train; inner OOF fits normal probability cutoff under 15 percent normal FP cap, outer held groups evaluate full pipeline',normal_false_alarm_cap=.15,bank_selection_policy='Train_mild_priority',candidate_C=[.1,1.,10.],development_used_only_after_selection=True,Test_used_for_selection=False,Test_input_sha256=sha(O/'test-inputs.frozen.json'),training_input_sha256=sha(O/'score-inputs.frozen.json'),code_sha256=sha(Path(__file__)),models_code_sha256=sha(B/'tools/error-enrichment/models.py'),production_replacement=False)
(O/'MODEL_SELECTION_PROTOCOL.frozen.json').write_text(json.dumps(protocol,indent=2))
def prob(model,rs):
 if model.model is None:
  return np.ones(len(rs)) if model.constant==2 else np.zeros(len(rs))
 p=model.model.predict_proba(model.scaler.transform([r['x'] for r in rs]));classes=list(model.model.classes_)
 return p[:,classes.index(2)] if 2 in classes else np.zeros(len(rs))
def predict_proba_records(model,rs):
 return [dict(id=r['id'],person=r['person'],group=r['group'],family=r['family'],target=r['target'],normal_probability=float(p)) for r,p in zip(rs,prob(model,rs))]
def grade(ps,t):return [dict(p,prediction=2. if p['normal_probability']>=t else 1.) for p in ps]
def folds(rs,n):
 g=[r['group'] for r in rs];return GroupKFold(n_splits=min(n,len(set(g)))).split(rs,groups=g)
def oof(rs,name,n=3):
 ps=[]
 for ti,vi in folds(rs,n):
  a=[rs[i] for i in ti];b=[rs[i] for i in vi];ps+=predict_proba_records(fit(a,name),b)
 return ps

def calibrate(ps):
 values=np.unique([p['normal_probability'] for p in ps]);cuts=np.r_[0,(values[:-1]+values[1:])/2,1.];choices=[]
 for t in cuts:
  predictions=grade(ps,float(t));m=metrics(predictions);mp=[p for p in predictions if p['id'] in mild];mm=metrics(mp);recall=mm['error_recall'] if mm else m['error_recall']
  rank=[0 if m['normal_false_alarm']<=.15 else 1,1-recall,1-m['error_recall'],m['normal_false_alarm'],float(t)]
  choices.append((rank,float(t),m,mm))
 return min(choices,key=lambda a:a[0])

buckets=collections.defaultdict(list)
for r in subset:
 for j,(n,g) in enumerate(zip(r['items'],r['levels'])):
  if n['eligible'] and g is not None and n['feature'] is not None:buckets[(r['exercise'],n['item'],n['bank'])].append(dict(id=r['id'],exercise=r['exercise'],index=j,x=n['feature'],target=g,person=r['person'],group=r['group'],family=r['family'],origin=r['origin'],split=r['split'],sources=r['sources']))
heads={};reports=[]
for key,rs in sorted(buckets.items()):
 tr=[r for r in rs if r['split']=='train'];va=[r for r in rs if r['split']=='validation'];assert not {r['group'] for r in tr}&{r['group'] for r in va};candidates=[]
 for C in [.1,1.,10.]:
  name='grade'+str(C);ps=[];base=[];fold_receipts=[]
  for ti,vi in folds(tr,4):
   a=[tr[i] for i in ti];b=[tr[i] for i in vi];inner=oof(a,name);_,t,im,imm=calibrate(inner);m=fit(a,name);outer=predict_proba_records(m,b);ps+=grade(outer,t);mean=float(np.average([r['target'] for r in a],weights=weights(a)));base += [dict(id=r['id'],group=r['group'],target=r['target'],prediction=mean) for r in b];fold_receipts.append(dict(held_groups=sorted({r['group'] for r in b}),fit_groups=sorted({r['group'] for r in a}),cutoff=t,cutoff_fit_ids=sorted(p['id'] for p in inner),outer_ids=sorted(r['id'] for r in b)))
  met=metrics(ps);mp=[p for p in ps if p['id'] in mild];mm=metrics(mp);recall=mm['error_recall'] if mm else 0.;valid=len(mp)==len(mild) and met['error_recall']>=.65 and met['normal_false_alarm']<=.15
  rank=[0 if valid else 1,1-recall+(len(mild)-len(mp))/len(mild),met['normal_false_alarm'],met['balanced_group_MAE']]
  candidates.append(dict(name=name,metrics=met,mild_metrics=mm,mild_coverage=len(mp),priority_rank=rank,predictions=ps,constant_baseline=metrics(base),folds=fold_receipts));print('CV',key,name,met,mm,flush=True)
 best=min(candidates,key=lambda c:c['priority_rank']);all_oof=oof(tr,best['name']);_,t,cm,cmm=calibrate(all_oof);base_model=fit(tr,best['name']);m=NormalThresholdRegressor(base_model,t);vp=predictions(m,va);gate=assess(best['predictions'],vp,best['constant_baseline'],None);counts=collections.Counter(r['target'] for r in tr)
 heads[key]=dict(model=m,key=key,feature_dimensions=len(tr[0]['x']),grade_support=dict(counts),trained_ids=[r['id'] for r in tr],candidate_gate_passed=gate['passed'],production_replacement=False)
 reports.append(dict(key=list(key),train_count=len(tr),validation_count=len(va),grade_counts=dict(counts),normal_groups=len({r['group'] for r in tr if r['target']==2}),error_groups=len({r['group'] for r in tr if r['target']<2}),source_families=dict(collections.Counter(r['family'] for r in tr)),status='trained_experimental',selected=best['name']+'_Train_OOF_calibrated',normal_probability_cutoff=t,group_cv=best['metrics'],constant_group_cv=best['constant_baseline'],mild_group_cv=best['mild_metrics'],mild_coverage=best['mild_coverage'],train_mild_priority_rank=best['priority_rank'],mild_Train_ids=sorted(mild),fixed_development_validation=metrics(vp),candidate_gate_passed=gate['passed'],runtime_replacement=False,acceptance=gate,selected_oof_predictions=best['predictions'],validation_predictions=vp,cutoff_Train_oof=all_oof,nested_CV_folds=best['folds'],candidate_cv=[{k:v for k,v in c.items() if k not in ['predictions','folds']} for c in candidates]))
 print('SELECTED',key,best['name'],'cutoff',t,'RANK',best['priority_rank'],'DEV',metrics(vp),flush=True)
joblib.dump(dict(schema='integrated-five-criterion-candidates-v1',heads=heads,labels_sha256=sha(O/'score-inputs.frozen.json'),runtime_replacement=False),SUB/'score-candidates.joblib');(SUB/'score-training.json').write_text(json.dumps(dict(heads=reports,completed=True,trained_heads=len(heads),gate_passed=sum(h['candidate_gate_passed'] for h in reports),final_test_used=False),indent=2));print('COMPLETE',len(heads),flush=True)
