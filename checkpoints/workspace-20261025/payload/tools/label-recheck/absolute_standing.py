"""Experimental calibration-free standing classifier; only supervised visual targets."""
import sys,json,os,collections
from pathlib import Path
import numpy as np,joblib
B=Path('/home/jayden/SeSACPAI-Mid');sys.path.insert(0,str(B/'tools/error-enrichment'))
from standing_state import load_frames,fit,probability,metric
from split_guard import POLICY
from train_scores import metrics
from sklearn.model_selection import GroupKFold
O=B/'output/label-verified-five-expanded-20261024'
def read(p):return json.loads(Path(p).read_text())
def save(n,v):(O/n).write_text(json.dumps(v,indent=2,allow_nan=False))
def angle(a,b,c):
 u=a-b;v=c-b;d=np.linalg.norm(u)*np.linalg.norm(v)
 return float(np.arccos(np.clip(np.dot(u,v)/d,-1,1))/np.pi) if d>1 else None
def feature(x):
 sides=[]
 for s,h,k,a in [(11,23,25,27),(12,24,26,28)]:
  if not np.isfinite(x[[s,h,k,a]]).all():continue
  lens=[np.linalg.norm(x[s]-x[h]),np.linalg.norm(x[h]-x[k]),np.linalg.norm(x[k]-x[a])]
  if min(lens)<2:continue
  sides.append([angle(x[h],x[k],x[a]),angle(x[s],x[h],x[k]),float(np.linalg.norm(x[s]-x[a])/sum(lens)),float(abs(x[a,1]-x[h,1])/(lens[1]+lens[2]))])
 if not sides:return None
 z=np.array(sides);return np.r_[z.mean(0),z.min(0)].tolist()
labels=read(B/'output/label-recheck-20261024/standing-states-reviewed.frozen.json');protected=set(read(POLICY)['protected_source_sha256']);rows=[];excluded=[]
for r in labels:
 if r['source_sha256'] in protected:continue
 xy,frames=load_frames(r);i=int(np.argmin(abs(frames-r['frame'])));f=feature(xy[i])
 if f is None or abs(int(frames[i])-r['frame'])/r['fps']>.061:excluded.append(r['id']);continue
 rows.append(dict(r,split='train' if r['review_id'].startswith('V') else r['split'],x=f,observed_frame=int(frames[i])))
tr=[r for r in rows if r['split']=='train'];va=[r for r in rows if r['split']=='validation'];assert not {r['person'] for r in tr}&{r['person'] for r in va}
save('absolute-standing-inputs.frozen.json',rows)
prior=read(B/'output/squat-return-20261023/feature-rows.json');lab={r['id']:r for r in read(B/'output/squat-return-20261023/labels.frozen.json')['labels']};attempts=[];holds=[]
for r in prior:
 d=lab[r['id']];xy,frames=load_frames(dict(review_id=Path(d['path']).stem));seq=xy[(frames>=d['first_frame'])&(frames<=d['last_frame'])];fs=[feature(x) for x in seq];valid=np.array([f is not None for f in fs]);hip=np.array([float(np.nanmean(x[[23,24],1])) if np.isfinite(x[[23,24],1]).any() else np.nan for x in seq])
 if len(seq)<10 or valid.mean()<.8 or not np.isfinite(hip).any():holds.append(dict(id=r['id'],reason='insufficient_leg_observation'));continue
 bottom=int(np.nanargmax(hip));tail=max(bottom+1,len(seq)-max(3,len(seq)//4));windows=[i for i in range(tail,len(seq)-2) if valid[i:i+3].all()]
 if not windows:holds.append(dict(id=r['id'],reason='post_bottom_tail_unobserved'));continue
 attempts.append(dict(r,xs=fs,valid=valid,windows=windows))
def rep_predict(rs,name,dims,final=None):
 out=[];models={}
 for r in rs:
  if final is None:
   if r['person'] not in models:models[r['person']]=fit([s for s in tr if s['person']!=r['person']],name,dims)
   m=models[r['person']]
  else:m=final
  good=np.flatnonzero(r['valid']);ps=np.full(len(r['xs']),np.nan);ps[good]=probability(m,[dict(x=r['xs'][i]) for i in good],dims);p=max(float(np.median(ps[i:i+3])) for i in r['windows']);out.append(dict(id=r['id'],person=r['person'],group=r['person'],target=r['target'],split=r['split'],prediction=1+p))
 return out
candidates=[]
for dims in [[0,1,2,3],[0,1,2,3,4,5,6,7]]:
 for name in ['log0.1','log1','log10','trees']:
  preds=rep_predict([r for r in attempts if r['split']=='train'],name,dims);cv=metrics(preds);candidates.append(dict(name=name,dims=dims,group_cv=cv));print('ABS_CV',name,dims,cv,flush=True)
def rank(r):
 c=r['group_cv'];return (not(c['error_recall']>=.65 and c['normal_false_alarm']<=.15),c['balanced_group_MAE'],len(r['dims']))
best=min(candidates,key=rank);m=fit(tr,best['name'],best['dims']);trainpred=rep_predict([r for r in attempts if r['split']=='train'],best['name'],best['dims']);valpred=rep_predict([r for r in attempts if r['split']=='validation'],best['name'],best['dims'],m)
joblib.dump(dict(model=m,dims=best['dims'],name=best['name'],training_ids=[r['id'] for r in tr],feature_schema='absolute_leg_angles_chain_lengths_v1',production_replacement=False),O/'absolute-standing-candidate.joblib')
save('absolute-standing-training.json',dict(selected_from_training_repetition_cv=True,candidates=candidates,selected=best,frame_train_n=len(tr),frame_validation_n=len(va),frame_validation=metric(va,probability(m,va,best['dims'])),frame_holds=excluded,original_attempt_cohort=len(prior),attempt_holds=holds,group_cv=metrics(trainpred),fixed_development_validation=metrics(valpred),predictions=trainpred+valpred,grade_zero_supported=False,test_used=False,production_replacement=False,limitations=['Same person-group development CV selects hyperparameters; not nested independent testing.','Standing frame targets do not establish full repetition failure ground truth.','Repetition severe-failure grade zero not supported.']))
print('ABS_COMPLETE',best,metrics(valpred),flush=True)
