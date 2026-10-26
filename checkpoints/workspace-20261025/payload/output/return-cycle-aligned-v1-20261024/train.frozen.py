import json,sys,collections
from pathlib import Path
import numpy as np,joblib
from sklearn.model_selection import GroupKFold,LeaveOneGroupOut
B=Path(__file__).resolve().parents[2];O=B/'output/return-cycle-aligned-v1-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from models import PositiveDeficitClassifier,MonotoneExtensionEnsemble
from train_scores import weights,metrics,predictions
from split_guard import assert_training_allowed
from acceptance import assess
read=lambda p:json.loads(p.read_text());key=('pushups','return_extension','observed_elbow_cycle_v1');rs=[]
for r in read(O/'score-inputs.frozen.json'):
 for j,(i,g) in enumerate(zip(r['items'],r['levels'])):
  if (r['exercise'],i['item'],i['bank'])==key and i['eligible'] and g is not None:rs.append(dict(id=r['id'],person=r['person'],group=r['group'],family=r['family'],exercise=r['exercise'],index=j,origin=r['origin'],sources=r['sources'],split=r['split'],x=i['feature'],target=g))
tr=[r for r in rs if r['split']=='train'];va=[r for r in rs if r['split']=='validation'];assert_training_allowed(tr)
def fit(a,C):
 assert_training_allowed(a);w=weights(a);normal=np.array([r['target']==2 for r in a])
 for mask in [normal,~normal]:
  if mask.any():w[mask]/=w[mask].sum()
 w*=len(w)/w.sum();m=PositiveDeficitClassifier(C).fit(MonotoneExtensionEnsemble.deficits([r['x'] for r in a]),[r['target'] for r in a],sample_weight=w)
 return m
def prs(m,a):return [dict(id=r['id'],person=r['person'],group=r['group'],family=r['family'],target=r['target'],normal_probability=float(p)) for r,p in zip(a,m.normal_probability([r['x'] for r in a]))]
def pred(ps,t):return [dict(p,prediction=2. if p['normal_probability']>=t else 1.) for p in ps]
def cutoff(ps):
 vals=np.unique([p['normal_probability'] for p in ps]);cs=[]
 for t in np.r_[0,(vals[1:]+vals[:-1])/2,1]:
  met=metrics(pred(ps,t));cs.append(([met['normal_false_alarm']>.05,1-met['error_recall'],met['normal_false_alarm'],met['balanced_group_MAE'],float(t)],float(t)))
 return min(cs)[1]
def bundle(a,C):
 members=[];ps=[];receipts=[]
 for ti,vi in LeaveOneGroupOut().split(a,groups=[r['group'] for r in a]):
  train=[a[i] for i in ti];held=[a[i] for i in vi];m=fit(train,C);members.append(m);ps+=prs(MonotoneExtensionEnsemble([m]),held);receipts.append(dict(fit_ids=[r['id'] for r in train],held_ids=[r['id'] for r in held],fit_groups=sorted({r['group'] for r in train}),held_groups=sorted({r['group'] for r in held})))
 return MonotoneExtensionEnsemble(members,cutoff(ps)),ps,receipts
candidates=[]
for C in [.1,1.,10.]:
 ps=[];receipts=[];baseline=[]
 for ti,vi in GroupKFold(4).split(tr,groups=[r['group'] for r in tr]):
  a=[tr[i] for i in ti];b=[tr[i] for i in vi];m,cal,rec=bundle(a,C);ps+=pred(prs(m,b),m.normal_threshold);mean=float(np.average([r['target'] for r in a],weights=weights(a)));baseline+=[dict(id=r['id'],group=r['group'],target=r['target'],prediction=mean) for r in b];receipts.append(dict(fit_ids=[r['id'] for r in a],outer_ids=[r['id'] for r in b],fit_groups=sorted({r['group'] for r in a}),held_groups=sorted({r['group'] for r in b}),inner_members=rec,cutoff=m.normal_threshold))
 met=metrics(ps);rank=[met['normal_false_alarm']>.05,met['error_recall']<.65,1-met['error_recall'],met['normal_false_alarm'],met['balanced_group_MAE']];candidates.append(dict(C=C,rank=rank,group_cv=met,selected_oof_predictions=ps,nested_CV_folds=receipts,constant_group_cv=metrics(baseline)));print('CV',C,rank,met,flush=True)
best=min(candidates,key=lambda c:c['rank']);(O/'TRAIN_SELECTION.frozen.json').write_text(json.dumps({k:v for k,v in best.items() if k not in ['selected_oof_predictions','nested_CV_folds']},indent=2));m,cal,rec=bundle(tr,best['C']);vp=predictions(m,va);gate=assess(best['selected_oof_predictions'],vp,best['constant_group_cv']);head=dict(model=m,key=key,feature_dimensions=8,grade_support=dict(collections.Counter(r['target'] for r in tr)),trained_ids=[r['id'] for r in tr],component_training=rec,candidate_gate_passed=gate['passed'],production_replacement=False)
report=dict(key=list(key),train_count=len(tr),validation_count=len(va),grade_counts=head['grade_support'],normal_groups=len({r['group'] for r in tr if r['target']==2}),error_groups=len({r['group'] for r in tr if r['target']<2}),source_families=dict(collections.Counter(r['family'] for r in tr)),status='trained_experimental',selected=f"monotone_extension_ensemble_C{best['C']}",normal_probability_cutoff=m.normal_threshold,group_cv=best['group_cv'],constant_group_cv=best['constant_group_cv'],candidate_gate_passed=gate['passed'],runtime_replacement=False,acceptance=gate,selected_oof_predictions=best['selected_oof_predictions'],validation_predictions=vp,fixed_development_validation=metrics(vp),nested_CV_folds=best['nested_CV_folds'],component_training=rec,cutoff_Train_oof=cal,candidate_cv=[{k:v for k,v in c.items() if k not in ['selected_oof_predictions','nested_CV_folds']} for c in candidates])
joblib.dump(dict(heads={key:head},labels_sha256=sha(O/'score-inputs.frozen.json')),O/'new-head.joblib');(O/'new-head-training.json').write_text(json.dumps(report,indent=2));print('SELECTED',report['selected'],'cutoff',m.normal_threshold,'DEV',metrics(vp),flush=True)
