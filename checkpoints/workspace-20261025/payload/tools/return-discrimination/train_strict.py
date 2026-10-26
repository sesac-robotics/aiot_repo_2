"""Nested group calibration on explicit initial-pose deficits."""
import sys,json,collections,copy
from pathlib import Path
import numpy as np,joblib
from sklearn.model_selection import GroupKFold
B=Path(__file__).resolve().parents[2];O=B/'output/return-discrimination-v2-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'))
from models import PositiveDeficitClassifier,NormalThresholdRegressor
from train_scores import fit,metrics,weights,predictions
from common import sha
from split_guard import assert_training_allowed
from acceptance import assess
read=lambda p:json.loads(p.read_text());rows=read(O/'score-inputs.frozen.json');mild=set(read(B/'output/mild-return-calibrated-20261024/MODEL_SELECTION_PROTOCOL.frozen.json')['mild_Train_ids']);rs=[]
for r in rows:
 for j,(i,g) in enumerate(zip(r['items'],r['levels'])):
  if i['bank']=='standing_support_extension_v1' and i['eligible'] and g is not None:rs.append(dict(id=r['id'],person=r['person'],group=r['group'],family=r['family'],exercise=r['exercise'],index=j,origin=r['origin'],sources=r['sources'],split=r['split'],x=i['feature'],target=g))
tr=[r for r in rs if r['split']=='train'];va=[r for r in rs if r['split']=='validation'];assert_training_allowed(tr)
def train(a,name,C):
 if name=='logistic':return fit(a,'grade'+str(C))
 assert_training_allowed(a);w=weights(a);normal=np.array([r['target']==2 for r in a]);
 for mask in [normal,~normal]:
  if mask.any():w[mask]/=w[mask].sum()
 w*=len(w)/w.sum();return PositiveDeficitClassifier(C).fit([r['x'] for r in a],[r['target'] for r in a],sample_weight=w)
def prob(m,a):
 if hasattr(m,'normal_probability'):return m.normal_probability([r['x'] for r in a])
 if m.model is None:return np.full(len(a),float(m.constant==2))
 classes=list(m.model.classes_);ps=m.model.predict_proba(m.scaler.transform([r['x'] for r in a]));return ps[:,classes.index(2)] if 2 in classes else np.zeros(len(a))
def prs(m,a):return [dict(id=r['id'],person=r['person'],group=r['group'],family=r['family'],target=r['target'],normal_probability=float(p)) for r,p in zip(a,prob(m,a))]
def pred(ps,t):return [dict(p,prediction=2. if p['normal_probability']>=t else 1.) for p in ps]
def folds(a,n):return GroupKFold(min(n,len({r['group'] for r in a}))).split(a,groups=[r['group'] for r in a])
def cutoff(ps):
 vals=np.unique([p['normal_probability'] for p in ps]);choices=[]
 for t in np.r_[0,(vals[1:]+vals[:-1])/2,1]:
  ps1=pred(ps,t);m=metrics(ps1);mm=metrics([p for p in ps1 if p['id'] in mild]);recall=mm['error_recall'] if mm else m['error_recall'];rank=[m['normal_false_alarm']>.05,1-recall,1-m['error_recall'],m['normal_false_alarm'],float(t)];choices.append((rank,float(t)))
 return min(choices)[1]
def oof(a,name,C):
 out=[]
 for ti,vi in folds(a,3):out+=prs(train([a[i] for i in ti],name,C),[a[i] for i in vi])
 return out
candidates=[]
for name in ['positive_deficit','logistic']:
 for C in [.1,1.,10.]:
  ps=[];receipt=[];baseline=[]
  for ti,vi in folds(tr,4):
   a=[tr[i] for i in ti];b=[tr[i] for i in vi];t=cutoff(oof(a,name,C));m=train(a,name,C);ps+=pred(prs(m,b),t);mean=float(np.average([r['target'] for r in a],weights=weights(a)));baseline+=[dict(id=r['id'],group=r['group'],target=r['target'],prediction=mean) for r in b];receipt.append(dict(fit_ids=[r['id'] for r in a],outer_ids=[r['id'] for r in b],cutoff=t,fit_groups=sorted({r['group'] for r in a}),held_groups=sorted({r['group'] for r in b})))
  met=metrics(ps);mp=[p for p in ps if p['id'] in mild];mm=metrics(mp);rank=[met['normal_false_alarm']>.05,met['error_recall']<.65 or len(mp)!=len(mild),1-mm['error_recall'],1-met['error_recall'],met['normal_false_alarm'],met['balanced_group_MAE']];c=dict(name=name,C=C,rank=rank,group_cv=met,mild_group_cv=mm,selected_oof_predictions=ps,nested_folds=receipt,constant_group_cv=metrics(baseline));candidates.append(c);print('CV',name,C,rank,met,flush=True)
best=min(candidates,key=lambda c:c['rank']);(O/'TRAIN_SELECTION.frozen.json').write_text(json.dumps({k:v for k,v in best.items() if k not in ['selected_oof_predictions','nested_folds']},indent=2));all_oof=oof(tr,best['name'],best['C']);t=cutoff(all_oof);m=train(tr,best['name'],best['C']);m.normal_threshold=t
if best['name']=='logistic':m=NormalThresholdRegressor(m,t)
vp=predictions(m,va);gate=assess(best['selected_oof_predictions'],vp,best['constant_group_cv'],None);key=('squats','standing_height_return','standing_support_extension_v1');head=dict(model=m,key=key,feature_dimensions=10,grade_support=dict(collections.Counter(r['target'] for r in tr)),trained_ids=[r['id'] for r in tr],candidate_gate_passed=gate['passed'],production_replacement=False)
report=dict(key=list(key),train_count=len(tr),validation_count=len(va),grade_counts=head['grade_support'],normal_groups=len({r['group'] for r in tr if r['target']==2}),error_groups=len({r['group'] for r in tr if r['target']<2}),source_families=dict(collections.Counter(r['family'] for r in tr)),status='trained_experimental',selected=f"{best['name']}_C{best['C']}_nested_Train_calibrated",normal_probability_cutoff=t,group_cv=best['group_cv'],constant_group_cv=best['constant_group_cv'],mild_group_cv=best['mild_group_cv'],mild_Train_ids=sorted(mild),candidate_gate_passed=gate['passed'],runtime_replacement=False,acceptance=gate,selected_oof_predictions=best['selected_oof_predictions'],validation_predictions=vp,fixed_development_validation=metrics(vp),nested_CV_folds=best['nested_folds'],candidate_cv=[{k:v for k,v in c.items() if k not in ['selected_oof_predictions','nested_folds']} for c in candidates],cutoff_Train_oof=all_oof)
joblib.dump(dict(heads={key:head},labels_sha256=sha(O/'score-inputs.frozen.json')),O/'new-head.joblib');(O/'new-head-training.json').write_text(json.dumps(report,indent=2));print('SELECTED',report['selected'],'cutoff',t,'DEV',metrics(vp),flush=True)
