"""Train native-condition encoders, then use probabilities as FEATURES only.
The main project grade still comes exclusively from independent project labels.
Provider Validation never trains these encoders. Source image FPS is not invented.
"""
from pathlib import Path
import sys,json,zipfile,io,re,collections,hashlib,joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score
from split_guard import assert_training_allowed
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/error-enrichment-20261024';JOINTS=[11,12,13,14,15,16,23,24,25,26,27,28];NAMES=['Left Shoulder','Right Shoulder','Left Elbow','Right Elbow','Left Wrist','Right Wrist','Left Hip','Right Hip','Left Knee','Right Knee','Left Ankle','Right Ankle'];TASKS={'plank_alignment':('플랭크','몸통과 엉덩이의 정렬 유지'),'pushup_descent':('푸시업','가슴의 충분한 이동'),'pushup_trunk':('푸시업','척추의 중립')}
def describe_common(a):
 a=np.asarray(a,float)
 if len(a)<3:return None
 valid=np.isfinite(a[:,[0,1,6,7]]).all((1,2));sh=a[:,[0,1]].mean(1);hip=a[:,[6,7]].mean(1);scale=np.linalg.norm(sh-hip,axis=1);valid&=scale>1
 if valid.mean()<.8:return None
 norm=(a-hip[:,None,:])/np.maximum(scale[:,None,None],1);norm[~valid]=np.nan;flat=norm.reshape(len(a),-1);f=[]
 for col in flat.T:
  good=col[np.isfinite(col)];f.extend(np.percentile(good,[5,25,50,75,95]).tolist() if len(good) else [np.nan]*5)
 f.extend(flat[np.linspace(0,len(flat)-1,8).astype(int)].flatten().tolist());return f
def fit(rows):
 assert_training_allowed(rows)
 pairs=collections.Counter((r['y'],r['person']) for r in rows);groups=collections.Counter(y for y,p in pairs);w=np.array([1/(groups[r['y']]*pairs[r['y'],r['person']]) for r in rows]);w*=len(w)/w.sum();m=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),ExtraTreesClassifier(n_estimators=160,min_samples_leaf=4,max_features=.7,n_jobs=1,random_state=241024));m.fit([r['x'] for r in rows],[r['y'] for r in rows],extratreesclassifier__sample_weight=w);return m
def main():
 rows=[];source=B/'output/five-exercise-comparison-20261021/aihub_documents'
 def collect(z,split,prefix=''):
  for n in z.namelist():
   if not n.endswith('.json') or n.endswith('-3d.json'):continue
   d=json.loads(z.read(n));meta=d.get('type_info',{});ex=meta.get('exercise');tasks=[(task,cond) for task,(exercise,cond) in TASKS.items() if exercise==ex]
   if not tasks:continue
   views=sorted({v for fr in d['frames'] for v,c in fr.items() if isinstance(c,dict) and c.get('pts')})
   for v in views:
    frames=[fr[v] for fr in d['frames'] if v in fr and fr[v].get('pts')];person=re.search(r'-(Z\d+)_[A-E]',frames[0]['img_key']).group(1);a=np.array([[[fr['pts'].get(j,{}).get(k,np.nan) for k in ['x','y']] for j in NAMES] for fr in frames]);x=describe_common(a)
    if x is None:continue
    for task,cond in tasks:
     target=next((c.get('value') for c in meta.get('conditions',[]) if c['condition']==cond),None)
     if type(target)is not bool:continue
     rows.append(dict(id=prefix+n+':'+v,task=task,person='aihub_'+person,split=split,x=x,y=int(target)))
 with zipfile.ZipFile(source/'training_body_new_220128.zip') as z:
  for n in z.namelist():
   if n.endswith('.zip'):
    with zipfile.ZipFile(io.BytesIO(z.read(n))) as sub:collect(sub,'train',n+':')
 with zipfile.ZipFile(source/'validation_body_01.zip') as z:collect(z,'validation')
 reports=[];models={}
 for task in TASKS:
  tr=[r for r in rows if r['task']==task and r['split']=='train'];va=[r for r in rows if r['task']==task and r['split']=='validation'];assert not {r['person'] for r in tr}&{r['person'] for r in va};g=np.array([r['person'] for r in tr]);ys=[];ps=[]
  for ti,vi in GroupKFold(4).split(tr,groups=g):
   m=fit([tr[i] for i in ti]);v=[tr[i] for i in vi];ys +=[r['y'] for r in v];ps+=m.predict([r['x'] for r in v]).tolist()
  m=fit(tr);models[task]=m;reports.append(dict(task=task,native_condition=TASKS[task][1],train_n=len(tr),validation_n=len(va),group_cv_balanced_accuracy=float(balanced_accuracy_score(ys,ps)),provider_validation_balanced_accuracy=float(balanced_accuracy_score([r['y'] for r in va],m.predict([r['x'] for r in va]))),train_people=sorted(set(g)),validation_people=sorted({r['person'] for r in va})));print(task,len(tr),flush=True)
 joblib.dump(dict(models=models,common_joints=JOINTS,schema='common12_torso_normalized_percentiles_order8_v1',outputs='native_condition_probabilities_only_not_project_grades'),O/'aihub-transfer-encoders.joblib');(O/'aihub-transfer-training.json').write_text(json.dumps(dict(tasks=reports,model_source='provided_2D_training_keypoints',raw_RGB_network_trained=False,provider_validation_not_trained=True,provider_boolean_not_converted_to_project_score=True,temporal_order_preserved_FPS_unknown=True),ensure_ascii=False,indent=2));(O/'aihub-transfer-inputs.frozen.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='x'} for r in rows],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
