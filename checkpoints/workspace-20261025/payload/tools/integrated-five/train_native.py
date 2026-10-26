"""Original provider tasks stay distinct from project criterion scores."""
import sys,collections,io,zipfile,re,time
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score,f1_score
from common import B,O,read,save,sha,status

def fit(X,y,groups):
 pairs=collections.Counter(zip(y,groups));per_grade=collections.Counter(label for label,group in pairs);weights=[1/(per_grade[label]*pairs[label,group]) for label,group in zip(y,groups)]
 return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),ExtraTreesClassifier(n_estimators=160,min_samples_leaf=3,max_features=.7,random_state=241024,n_jobs=2)).fit(np.asarray(X),np.asarray(y),extratreesclassifier__sample_weight=np.asarray(weights))
def metric(y,p):return dict(n=len(y),balanced_accuracy=float(balanced_accuracy_score(y,p)),macro_f1=float(f1_score(y,p,average='macro',zero_division=0)))
def experiment(name,X,y,groups,ids,validation=None):
 X=np.asarray(X);y=np.asarray(y);groups=np.asarray(groups);report=dict(train_n=len(y),train_groups=len(set(groups)),class_counts=dict(collections.Counter(map(str,y))),expert_ground_truth=False)
 if len(set(y))<2 or min(collections.Counter(y).values())<4 or len(set(groups))<3:return dict(report,status='insufficient_classes_or_groups')
 ps=[];ys=[];folds=[]
 for tr,va in GroupKFold(n_splits=min(4,len(set(groups)))).split(X,y,groups):
  assert not set(groups[tr])&set(groups[va]);m=fit(X[tr],y[tr],groups[tr]);p=m.predict(X[va]);ys.extend(y[va].tolist());ps.extend(p.tolist());folds.append(dict(train_groups=sorted(set(groups[tr])),validation_groups=sorted(set(groups[va])),metrics=metric(y[va],p)))
 m=fit(X,y,groups);path=O/'native-models'/(name+'.joblib');path.parent.mkdir(exist_ok=True);joblib.dump(dict(model=m,task=name,semantics='provider_native_label_not_project_score',train_ids=ids,train_groups=sorted(set(groups))),path)
 result=dict(report,status='trained_experimental',cross_validation=metric(ys,ps),folds=folds,model=str(path),sha256=sha(path))
 if validation:
  vx,vy,vg=validation;assert not set(groups)&set(vg);result['provider_validation']=metric(vy,m.predict(vx))
 return result

def qevd():
 cache=B/'mediapipe-pilot/qevd_coach/artifacts_v2/heavy/features';items=read(B/'mediapipe-pilot/qevd_coach/catalog.json')['items'];rs=[];xs=[];excluded=collections.Counter()
 for r in items:
  if r['exercise'] not in ['squats','pushups','plank','shoulder','ytw']:continue
  if r['split']!='train':excluded['reserved_provider_'+r['split']]+=1;continue
  p=cache/(r['id']+'.npz')
  if not p.exists():excluded['no_cache']+=1;continue
  z=np.load(p);x=z['X']
  if not x.size:excluded['insufficient_observation']+=1;continue
  rs.append(r);xs.append(x)
 reports=[]
 for ex in sorted({r['exercise'] for r in rs}):
  errors=sorted({e for r in rs if r['exercise']==ex for e in r['errors']})
  for k,error in enumerate(errors):
   selected=[i for i,r in enumerate(rs) if r['exercise']==ex and (r['reference'] or error in r['errors'])];r=[rs[i] for i in selected]
   report=experiment(f'qevd-{ex}-{k:02}',[xs[i] for i in selected],[int(error in a['errors']) for a in r],['qevd_'+a['worker'] for a in r],[a['id'] for a in r]);reports.append(dict(exercise=ex,error=error,**report));save('qevd-native-training.json',dict(tasks=reports,excluded=dict(excluded),input_count=len(rs),source_policy='Only official train rows with explicit reference or explicit matching error. Missing error is not normal. Official test not opened.'))
   status('qevd_native',exercise=ex,error=error,rows=len(r))
 save('qevd-native-inputs.json',[dict(id=r['id'],person='qevd_'+r['worker'],exercise=r['exercise'],path=r['path'],split='train',cache=str(cache/(r['id']+'.npz'))) for r in rs])

def rehab():
 p=B/'mediapipe-pilot/correction';z=np.load(p/'training-features.npz');rows=read(p/'sample-manifest.json');tr=z['splits']=='train';va=z['splits']=='validation';y=z['y'];reports=[]
 for task in ['normal_error','error_type']:
  yy=np.where(y=='NORMAL','NORMAL','ERROR') if task=='normal_error' else y
  result=experiment('rehab26-'+task,z['X'][tr],yy[tr],[r['person'] for r,m in zip(rows,tr) if m],[f"{r['video_id']}_{r['first_frame']}_{r['last_frame']}" for r,m in zip(rows,tr) if m],validation=(z['X'][va],yy[va],[r['person'] for r,m in zip(rows,va) if m]));reports.append(dict(task=task,**result));save('rehab26-native-training.json',dict(tasks=reports,feature_sha256=sha(p/'training-features.npz'),historical_test_evaluated=False,note='Original provider rehabilitation task. No conversion of error type to all-zero project scores.'))

def aihub():
 source=B/'output/five-exercise-comparison-20261021/aihub_documents';targets={'푸시업':'pushups','플랭크':'plank','Y - Exercise':'Y_only'};samples=[];mapping={'Left Shoulder':11,'Right Shoulder':12,'Left Elbow':13,'Right Elbow':14,'Left Wrist':15,'Right Wrist':16,'Left Hip':23,'Right Hip':24,'Left Knee':25,'Right Knee':26,'Left Ankle':27,'Right Ankle':28,'Nose':0,'Neck':1,'Left Eye':2,'Right Eye':3,'Left Ear':4,'Right Ear':5,'Left Palm':6,'Right Palm':7,'Back':8,'Waist':9,'Left Foot':10,'Right Foot':17}
 def collect(z,split,prefix=''):
  for member in z.namelist():
   if not member.endswith('.json') or member.endswith('-3d.json'):continue
   import json
   d=json.loads(z.read(member));meta=d.get('type_info',{});ex=targets.get(meta.get('exercise'))
   if ex is None:continue
   views=sorted({v for fr in d['frames'] for v,c in fr.items() if isinstance(c,dict) and c.get('pts')})
   for view in views:
    seq=[];imgkeys=[]
    for fr in d['frames']:
     c=fr.get(view,{});pts=c.get('pts');
     if not pts:continue
     a=np.array([[pts.get(n,{}).get('x',np.nan),pts.get(n,{}).get('y',np.nan)] for n in mapping],float);seq.append(a);imgkeys.append(c['img_key'])
    a=np.asarray(seq)
    if len(a)<3:continue
    hip=(a[:,6]+a[:,7])/2;torso=np.linalg.norm((a[:,0]+a[:,1])/2-hip,axis=1)
    if np.any(torso<=1):continue
    norm=(a-hip[:,None,:])/torso[:,None,None];flat=norm.reshape(len(a),-1);features=[]
    for col in flat.T:
     good=col[np.isfinite(col)];features.extend(np.percentile(good,[5,25,50,75,95]).tolist() if len(good) else [np.nan]*5)
    features.extend(flat[np.linspace(0,len(a)-1,8).astype(int)].flatten().tolist())
    person=re.search(r'-(Z\d+)_[A-E]',imgkeys[0]);
    if not person:raise ValueError('AIHub identity unresolved')
    for condition in meta.get('conditions',[]):
     if type(condition.get('value')) is not bool or '엄지' in condition['condition']:continue
     samples.append(dict(id=prefix+member+':'+view,exercise=ex,person='aihub_'+person.group(1),split=split,condition=condition['condition'],target=int(condition['value']),features=features,image_references=imgkeys))
 with zipfile.ZipFile(source/'training_body_new_220128.zip') as z:
  for n in z.namelist():
   if n.endswith('.zip'):
    with zipfile.ZipFile(io.BytesIO(z.read(n))) as sub:collect(sub,'train',n+':')
 with zipfile.ZipFile(source/'validation_body_01.zip') as z:collect(z,'validation')
 save('aihub-native-inputs.json',[{k:v for k,v in r.items() if k!='features'} for r in samples]);reports=[]
 for idx,(ex,cond) in enumerate(sorted({(r['exercise'],r['condition']) for r in samples})):
  tr=[r for r in samples if r['split']=='train' and (r['exercise'],r['condition'])==(ex,cond)];va=[r for r in samples if r['split']=='validation' and (r['exercise'],r['condition'])==(ex,cond)];assert not {r['person'] for r in tr}&{r['person'] for r in va}
  import hashlib
  taskname=f'aihub-{ex}-joints24-'+hashlib.sha256(cond.encode()).hexdigest()[:8]
  result=experiment(taskname,[r['features'] for r in tr],[r['target'] for r in tr],[r['person'] for r in tr],[r['id'] for r in tr],validation=([r['features'] for r in va],[r['target'] for r in va],[r['person'] for r in va]) if va else None);reports.append(dict(exercise=ex,condition=cond,**result));save('aihub-native-training.json',dict(tasks=reports,source_sha256=sha(source/'training_body_new_220128.zip'),semantics='Provider 2D joints and native boolean conditions, not raw-image network or project 100-point score. Y-only does not supervise T/W or sequence.',raw_training_images_matched=False,validation_kept_separate=True,unsupported_conditions=[dict(condition='엄지손가락 하늘방향',reason='Thumb direction is not observable from supplied 2D keypoints; needs original images or hand landmarks.')],feature_schema='24 provider keypoints, torso-normalized spatial statistics and ordered time samples'))
  status('aihub_native',exercise=ex,condition=cond,train=len(tr),validation=len(va))
if __name__=='__main__':
 for name,fn in [('qevd',qevd),('rehab26',rehab),('aihub',aihub)]:
  status('native_start',source=name);fn()
 status('native_complete')
