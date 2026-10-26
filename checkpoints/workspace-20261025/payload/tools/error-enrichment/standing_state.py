"""Learned standing-height state from independently annotated visual frames.
A flexed frame is not an unsuccessful repetition; temporal scoring is separate.
"""
from pathlib import Path
import sys,json,collections,hashlib,joblib,os
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score,confusion_matrix
from split_guard import assert_training_allowed
B=Path('/home/jayden/SeSACPAI-Mid');OBS=B/'output/error-enrichment-20261024';O=Path(os.environ.get('POSTURE_RUN_OUTPUT',str(OBS)));sys.path.insert(0,str(B/'tools/integrated-five'));from team_features import height_features
D=B/'data/error-enrichment-20261024/annotations'
def read(p):return json.loads(Path(p).read_text())
def save(n,v):(O/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.item() if isinstance(a,np.generic) else a.tolist()))
def cache_for(r):
 e=r['review_id']
 if e.startswith('E'):return OBS/'observations'/e/'primary.npz'
 if e.startswith('V'):return B/'output/integrated-five-20261024/team-observations'/e/'side.npz'
 return B/'output/squat-return-20261023'/(e+'-heavy.npz')
def load_frames(r):
 z=np.load(cache_for(r));return z['image_xy'],z['frame_numbers']
def knee_angle(x,h,k,a):
 if not np.isfinite(x[[h,k,a]]).all():return None
 u=x[h]-x[k];v=x[a]-x[k];den=np.linalg.norm(u)*np.linalg.norm(v)
 return float(np.degrees(np.arccos(np.clip(np.dot(u,v)/den,-1,1)))) if den>1 else None
def add_leg_reference(ref,x):
 ref['leg_reference']={}
 for h,k,a in [(23,25,27),(24,26,28)]:
  angle=knee_angle(x,h,k,a)
  if angle is not None:ref['leg_reference'][str(h)]=dict(knee_angle=angle,hip_ankle_height=float(x[a,1]-x[h,1]))
 return ref
def frame_x(x,ref):
 if not np.isfinite(x[[11,12,23,24]]).all():return None
 h=x[[23,24]].mean(0);s=x[[11,12]].mean(0);scale=ref['scale'];horizontal=abs(h[0]-ref['center_x'])/scale;ratio=np.linalg.norm(s-h)/scale
 if horizontal>.45 or not .4<=ratio<=2.5:return None
 knee=[];leg=[]
 for hidx,k,a in [(23,25,27),(24,26,28)]:
  initial=ref.get('leg_reference',{}).get(str(hidx));angle=knee_angle(x,hidx,k,a)
  if initial is not None and angle is not None:
   knee.append((initial['knee_angle']-angle)/180);leg.append((initial['hip_ankle_height']-(x[a,1]-x[hidx,1]))/scale)
 if not knee:return None
 return [(h[1]-ref['hip_y'])/scale,(s[1]-ref['shoulder_y'])/scale,ratio,float(np.mean(knee)),float(np.mean(leg))]
def weights(rs):
 c=collections.Counter((r['target'],r['person']) for r in rs);g=collections.Counter(t for t,p in c);w=np.array([1/(g[r['target']]*c[r['target'],r['person']]) for r in rs]);return w*len(w)/w.sum()
def model(name):
 if name.startswith('log'):return make_pipeline(StandardScaler(),LogisticRegression(C=float(name[3:]),max_iter=1000,random_state=241024))
 if name.startswith('svm'):return make_pipeline(StandardScaler(),SVC(C=float(name[3:]),gamma='scale',probability=True,random_state=241024))
 return ExtraTreesClassifier(n_estimators=160,min_samples_leaf=3,max_features=1.,random_state=241024,n_jobs=1)
def fit(rs,name,dims):
 assert_training_allowed(rs)
 m=model(name);X=[[r['x'][i] for i in dims] for r in rs];y=[r['target'] for r in rs];w=weights(rs)
 if name.startswith('log'):m.fit(X,y,logisticregression__sample_weight=w)
 elif name.startswith('svm'):m.fit(X,y,svc__sample_weight=w)
 else:m.fit(X,y,sample_weight=w)
 return m
def probability(m,rs,dims):return m.predict_proba([[r['x'][i] for i in dims] for r in rs])[:,list(m.classes_).index(1)]
def metric(rs,ps):
 y=np.array([r['target'] for r in rs]);p=np.array(ps)>=.5
 return dict(n=len(y),people=len({r['person'] for r in rs}),balanced_accuracy=float(balanced_accuracy_score(y,p)),standing_recall=float(p[y==1].mean()) if (y==1).any() else None,flexed_recall=float((~p[y==0]).mean()) if (y==0).any() else None,confusion_matrix=confusion_matrix(y,p,labels=[0,1]).tolist())
def prepare():
 label_path=Path(os.environ['POSTURE_STATE_LABELS']) if os.environ.get('POSTURE_STATE_LABELS') else next(D/n for n in ['standing-states-final.frozen.json','standing-states-hard-reviewed.frozen.json','standing-states-combined.frozen.json'] if (D/n).exists())
 labels=read(label_path);groups=collections.defaultdict(list)
 if os.environ.get('POSTURE_RUN_OUTPUT'):
  from split_guard import POLICY
  policy=read(POLICY);labels=[dict(r,split='train') if r['review_id'].startswith('V') else r for r in labels if r['source_sha256'] not in policy['protected_source_sha256']]
 for r in labels:groups[r['review_id']].append(r)
 result=[];refs={};missing=[]
 for e,rs in groups.items():
  anchors=[r for r in rs if r['target']==1]
  if not anchors:
   missing.append(dict(review_id=e,reason='no_visually_verified_standing_calibration'));continue
  xy,frames=load_frames(rs[0]);standing=min(anchors,key=lambda r:r['frame']);index=int(np.argmin(abs(frames-standing['frame'])));x=xy[index];ref=height_features.reference(np.repeat(x[None],3,axis=0))
  if ref is None:missing.append(dict(review_id=e,reason='standing_calibration_unobserved'));continue
  ref=add_leg_reference(ref,x);ref.update(calibration_frame=int(frames[index]),source_sha256=standing['source_sha256'],person=standing['person'],split=standing['split']);refs[e]=ref
  for r in rs:
   ix=int(np.argmin(abs(frames-r['frame'])));delta=abs(int(frames[ix])-r['frame'])/r['fps'];a=frame_x(xy[ix],ref)
   if delta>.061 or a is None:missing.append(dict(id=r['id'],reason='frame_or_required_torso_not_observed'));continue
   result.append(dict(r,x=a,observed_frame=int(frames[ix]),cache=str(cache_for(r)),calibration_precedes_frame=frames[ix]>=ref['calibration_frame']))
 save('standing-state-inputs.frozen.json',result);save('standing-calibrations.json',refs);save('standing-state-exclusions.json',missing);return result,refs

def main():
 rows,refs=prepare();tr=[r for r in rows if r['split']=='train'];va=[r for r in rows if r['split']=='validation'];assert not {r['person'] for r in tr}&{r['person'] for r in va};groups=np.array([r['person'] for r in tr]);cv=[];splits=list(GroupKFold(n_splits=4).split(tr,groups=groups))
 for dims in [[0],[0,1,2],[0,3],[0,3,4],[0,1,2,3,4]]:
  for name in ['log0.1','log1','log10','svm1','svm10','trees']:
   allrs=[];allp=[]
   for ti,vi in splits:
    t=[tr[i] for i in ti];v=[tr[i] for i in vi];m=fit(t,name,dims);allrs+=v;allp+=probability(m,v,dims).tolist()
   cv.append(dict(name=name,dims=dims,metrics=metric(allrs,allp)))
 best=max(cv,key=lambda r:(r['metrics']['balanced_accuracy'],-len(r['dims'])));m=fit(tr,best['name'],best['dims']);vp=probability(m,va,best['dims']);bundle=dict(model=m,dims=best['dims'],name=best['name'],training_ids=[r['id'] for r in tr],training_people=sorted({r['person'] for r in tr}),feature_schema='relative_hip_height_optional_shoulder_height_and_torso_ratio_v1',production_replacement=False,frame_state_not_repetition_failure=True);joblib.dump(bundle,O/'standing-state-candidate.joblib');save('standing-state-training.json',dict(selected=best,candidates=cv,train_frames=len(tr),validation_frames=len(va),validation=metric(va,vp),validation_predictions=[dict(id=r['id'],person=r['person'],target=r['target'],probability=float(p)) for r,p in zip(va,vp)],final_test_used=False,production_replacement=False,expert_ground_truth=False));print('STATE_TRAINED',best,'validation',metric(va,vp),flush=True)
 # Evaluate the NEW state model on SAME previously reviewed return repetitions.
 lab={r['id']:r for r in read(B/'output/squat-return-20261023/labels.frozen.json')['labels']};prior=read(B/'output/squat-return-20261023/feature-rows.json');preds=[];ex=[];cvmodels={}
 for r in prior:
  d=lab[r['id']];source=Path(d['path']).stem;ref=refs.get(source)
  if ref is None:ex.append(dict(id=r['id'],reason='no_reviewed_preceding_calibration'));continue
  if d['first_frame']<=ref['calibration_frame']:ex.append(dict(id=r['id'],reason='attempt_before_calibration'));continue
  xy,frames=load_frames(dict(review_id=source));ix=np.flatnonzero((frames>=d['first_frame'])&(frames<=d['last_frame']));seq=xy[ix];xs=[frame_x(a,ref) for a in seq];valid=np.array([a is not None for a in xs]);
  if len(ix)<10 or valid.mean()<.8:ex.append(dict(id=r['id'],reason='required_torso_observation_insufficient'));continue
  drop=np.array([a[0] if a else np.nan for a in xs]);bottom=int(np.nanargmax(drop));tail=max(bottom+1,len(seq)-max(3,len(seq)//4));windows=[j for j in range(tail,len(seq)-2) if valid[j:j+3].all()]
  if not windows:ex.append(dict(id=r['id'],reason='post_descent_return_unobserved'));continue
  if r['split']=='train':
   if r['person'] not in cvmodels:cvmodels[r['person']]=fit([s for s in tr if s['person']!=r['person']],best['name'],best['dims'])
   mm=cvmodels[r['person']]
  else:mm=m
  pp=np.full(len(seq),np.nan);good=np.flatnonzero(valid);pp[good]=probability(mm,[dict(x=xs[j]) for j in good],best['dims']);p=max(float(np.median(pp[j:j+3])) for j in windows);preds.append(dict(id=r['id'],person=r['person'],group=r['person'],split=r['split'],target=r['target'],prediction=1+p,standing_probability=p,calibration_frame=ref['calibration_frame'],cross_person_state_training=r['split']=='train'))
 from train_scores import metrics
 report=dict(state_selected_from_frame_training_cv_only=True,group_cv=metrics([r for r in preds if r['split']=='train']),fixed_development_validation=metrics([r for r in preds if r['split']=='validation']),predictions=preds,excluded=ex,feature_schema='learned_standing_state_then_three_observation_temporal_return_v1',grade_zero_supported=False,final_test_used=False,production_replacement=False,warning='This is development evidence; calibration is visually identified before each evaluated attempt. No flexed frame was called a failed repetition. Training-side state predictions leave the assessed person out, but model hyperparameter selection is development CV rather than nested independent testing.')
 save('standing-return-state-comparison.json',report);print('RETURN_COMPARISON',report['group_cv'],report['fixed_development_validation'],flush=True)
if __name__=='__main__':main()
