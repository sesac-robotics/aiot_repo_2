from pathlib import Path
import json,collections,hashlib
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesClassifier
from standing_state import make_reference,features,CORE_NAMES,KNEE_NAMES,VERSION
O=Path(__file__).resolve().parent;S=O.parent/'squat-return-20261023';labels=json.loads((O/'labels.frozen.json').read_text())['rows'];cache={};rows=[]
for r in labels:
 if r['source'] not in cache:
  z=np.load(S/(r['source']+'-heavy.npz'));ref=make_reference(np.load(O/(r['source']+'-reference.npz'))['image_xy']);cache[r['source']]=(z,ref)
 z,ref=cache[r['source']];idx=np.flatnonzero(z['frame_numbers']==r['frame']);f=features(z['image_xy'][idx[0]],ref,int(z['width']),int(z['height']))
 if f:rows.append(dict(**r,features=f))
(O/'feature-rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
def fit(rs,bank):
 rs=[r for r in rs if r['features'][bank] is not None];groups=collections.Counter((r['person'],r['label']) for r in rs)
 m=ExtraTreesClassifier(n_estimators=160,min_samples_leaf=2,max_features=1.,random_state=23,n_jobs=2).fit([r['features'][bank] for r in rs],[r['label'] for r in rs],sample_weight=[1/groups[(r['person'],r['label'])] for r in rs]);return m

def predict(m,rs,bank):
 rs=[r for r in rs if r['features'][bank] is not None and r['original_review_frame'] not in r['reference_frames']];p=m.predict_proba([r['features'][bank] for r in rs])[:,list(m.classes_).index(1)];return [dict(id=r['id'],person=r['person'],label=r['label'],probability=float(v)) for r,v in zip(rs,p)]
def metrics(ps):
 return dict(n=len(ps),standing_n=sum(p['label']==1 for p in ps),nonstanding_n=sum(p['label']==0 for p in ps),standing_missed=sum(p['label']==1 and p['probability']<.5 for p in ps),nonstanding_as_standing=sum(p['label']==0 and p['probability']>=.5 for p in ps),accuracy=float(np.mean([(p['probability']>=.5)==p['label'] for p in ps])))
tr=[r for r in rows if r['split']=='train'];held=[r for r in rows if r['split']=='validation'];report={};models={}
for bank in ['core','knee']:
 oof=[];folds=[]
 for person in sorted({r['person'] for r in tr}):
  a=[r for r in tr if r['person']!=person];b=[r for r in tr if r['person']==person];ps=predict(fit(a,bank),b,bank);oof+=ps;folds.append(dict(person=person,metrics=metrics(ps)))
 models[bank]=fit(tr,bank);ps=predict(models[bank],held,bank);report[bank]=dict(group_cv=metrics(oof),fixed_validation=metrics(ps),folds=folds,oof_predictions=oof,validation_predictions=ps)
 print(bank,json.dumps({k:v for k,v in report[bank].items() if 'predictions' not in k},ensure_ascii=False),flush=True)
labelsha=hashlib.sha256((O/'labels.frozen.json').read_bytes()).hexdigest();bundle=dict(version=VERSION,models=models,feature_names=dict(core=CORE_NAMES,knee=CORE_NAMES+KNEE_NAMES),threshold=.5,training_people=sorted({r['person'] for r in tr}),validation_people=['cfrep_w2'],label_sha256=labelsha,labels_are_state_not_failed_repetitions=True)
joblib.dump(bundle,O/'model.joblib');result=dict(version=VERSION,items=report,label_sha256=labelsha,model_sha256=hashlib.sha256((O/'model.joblib').read_bytes()).hexdigest(),independent_final_test=False,expert_validated=False,raw_labels=len(labels),eligible_labels=len(rows),runtime_deployed=False);(O/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
