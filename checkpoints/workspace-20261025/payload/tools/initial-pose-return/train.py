"""Train explicit ready-state/phase AI from reviewed Train frames, grouped by person."""
import json,sys,datetime,collections,os
from pathlib import Path
from fractions import Fraction
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score,confusion_matrix
B=Path('/home/jayden/SeSACPAI-Mid');O=Path(os.environ.get('INITIAL_POSE_RUN',str(B/'output/initial-pose-return-v4-20261024')));sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import assert_training_allowed,POLICY
from recognizer import pose,relative,READY_MEANING,ReturnTracker
labels=json.loads((O/'state-labels-v3.frozen.json').read_text())['labels'];inventory={r['id']:r for r in json.loads((B/'output/r2d2-review-20261024/inventory.json').read_text())};data=[];held=[];caches={}
for r in labels:
 s=inventory.get(r['source_id']);p=Path(r['cache']) if r.get('cache') else B/'output/integrated-five-20261024/team-observations'/r['source_id']/'side.npz'
 if r['source_id'] not in caches:
  source=r.get('source_path') or s['copy'];fps=r.get('fps') or s['fps']
  assert sha(source)==r['source_sha256'];z=np.load(p);m=json.loads(p.with_suffix('.json').read_text());assert m['config']['source_sha256']==r['source_sha256'] and str(z['fingerprint'])==m['fingerprint'];caches[r['source_id']]=(z,z['frame_numbers']/float(Fraction(fps)),str(p),sha(p))
 z,t,cp,digest=caches[r['source_id']];n=int(np.argmin(abs(t-r['review_time'])));x=pose(r['exercise'],z['image_xy'][n])
 if abs(t[n]-r['review_time'])>.075 or x is None:held.append(dict(id=r['id'],reason='required_pose_or_exact_frame_unobserved'));continue
 data.append(dict(r,x=x.tolist(),observed_time=float(t[n]),observed_frame=int(z['frame_numbers'][n]),cache=cp,cache_sha256=digest))
refs={}
for r in sorted(data,key=lambda r:r['observed_time']):
 if r['target']==1 and r['source_id'] not in refs:refs[r['source_id']]=dict(id=r['id'],x=r['x'],time=r['observed_time'],person=r['person'])
for r in data:
 ref=refs.get(r['source_id']);r['reference_id']=ref['id'] if ref else None;r['relative_x']=relative(r['x'],ref['x']).tolist() if ref and r['observed_time']>=ref['time'] else None
assert_training_allowed(data);(O/'state-inputs.frozen.json').write_text(json.dumps(data,indent=2));(O/'state-exclusions.json').write_text(json.dumps(held,indent=2));(O/'initial-references.frozen.json').write_text(json.dumps(refs,indent=2))
temporal=[]
if (O/'plank-temporal-development.frozen.json').exists():
 for r in json.loads((O/'plank-temporal-development.frozen.json').read_text())['windows']:
  assert sha(r['cache'])==r['cache_sha256'];z=np.load(r['cache']);meta=json.loads(Path(r['cache']).with_suffix('.json').read_text());assert meta['config']['source_sha256']==r['source_sha256'] and str(z['fingerprint'])==meta['fingerprint'];ts=z['frame_numbers']/r['fps'];ix=np.flatnonzero((ts>=r['start'])&(ts<=r['end']));temporal.append(dict(r,times=ts[ix].tolist(),vectors=[pose('plank',z['image_xy'][i]) for i in ix]))
 assert_training_allowed(temporal)
def make(name):
 if name.startswith('svm'):return make_pipeline(StandardScaler(),SVC(C=float(name[3:]),probability=True,random_state=241024))
 return make_pipeline(StandardScaler(),LogisticRegression(C=.3 if name=='log0.3' else 3,max_iter=1500,random_state=241024)) if name.startswith('log') else ExtraTreesClassifier(n_estimators=128,min_samples_leaf=2,max_features=1.,random_state=241024,n_jobs=1)
def fit(rs,name,field,target):
 assert_training_allowed(rs);m=make(name);counts=collections.Counter((r[target],r['person']) for r in rs);ys=collections.Counter(y for y,p in counts);w=np.array([1/(ys[r[target]]*counts[r[target],r['person']]) for r in rs]);w*=len(w)/w.sum() if name.startswith('svm') else 1;kw={'logisticregression__sample_weight':w} if name.startswith('log') else {'svc__sample_weight':w} if name.startswith('svm') else {'sample_weight':w};m.fit([r[field] for r in rs],[r[target] for r in rs],**kw);return m
bundle={};reports=[]
for ex in READY_MEANING:
 for field in ['x','relative_x']:
  for target in (['target','phase'] if ex=='ytw' else ['target']):
   rs=[r for r in data if r['exercise']==ex and r[field] is not None and r[target] is not None];key=(ex,field,target);groups=np.array([r['person'] for r in rs]);classes=sorted({r[target] for r in rs});support={str(y):len({r['person'] for r in rs if r[target]==y}) for y in classes}
   if len(classes)<2 or min(support.values())<2:reports.append(dict(key=key,status='unsupported_missing_state_class_groups',class_groups=support));continue
   candidates=[]
   names=['log0.3','log3','trees']+(['svm1','svm10'] if ex=='plank' else [])
   for name in names:
    preds=[];temporal_predictions=[];valid=True
    for ti,vi in LeaveOneGroupOut().split(rs,groups=groups):
     tr=[rs[i] for i in ti];va=[rs[i] for i in vi if rs[i]['id']!=rs[i]['reference_id']]
     if set(r[target] for r in tr)!=set(classes):valid=False;break
     if not va:continue
     m=fit(tr,name,field,target);p=m.predict([r[field] for r in va]);prob=m.predict_proba([r[field] for r in va]);preds.extend(dict(id=r['id'],source_id=r['source_id'],person=r['person'],time=r['observed_time'],target=r[target],prediction=a.item() if isinstance(a,np.generic) else a,probability={str(c):float(q) for c,q in zip(m.classes_,pr)}) for r,a,pr in zip(va,p,prob))
     if ex=='plank' and field=='x' and target=='target':
      held_people=set(groups[vi])
      for window in [w for w in temporal if w['person'] in held_people]:
       tracker=ReturnTracker('plank')
       for stamp,vector in zip(window['times'],window['vectors']):tracker.update(stamp,None if vector is None else float(m.predict([vector])[0]))
       temporal_predictions.append(dict(source_id=window['source_id'],person=window['person'],target=window['target'],initial_ready_observed=tracker.ready,initial_ready_time=tracker.calibration_time))
    if not valid or not preds:continue
    y=[p['target'] for p in preds];a=[p['prediction'] for p in preds];met=dict(n=len(y),balanced_accuracy=float(balanced_accuracy_score(y,a)),confusion_matrix=confusion_matrix(y,a,labels=classes).tolist(),classes=classes)
    if target=='target':met.update(ready_n=sum(yv==1 for yv in y),nonready_n=sum(yv==0 for yv in y),ready_missed=sum(p['target']==1 and p['prediction']==0 for p in preds),nonready_called_ready=sum(p['target']==0 and p['prediction']==1 for p in preds))
    tm=None
    if temporal_predictions:
     bad=[p for p in temporal_predictions if p['target']==0];good=[p for p in temporal_predictions if p['target']==1];tm=dict(windows=len(temporal_predictions),nonready_windows=len(bad),ready_windows=len(good),false_initial_ready=sum(p['initial_ready_observed'] for p in bad),missed_initial_ready=sum(not p['initial_ready_observed'] for p in good),predictions=temporal_predictions);tm['passed']=bool(bad and good and tm['false_initial_ready']==0 and tm['missed_initial_ready']==0)
    candidates.append(dict(name=name,metrics=met,predictions=preds,temporal_development=tm))
   if not candidates:reports.append(dict(key=key,status='unsupported_group_cv'));continue
   best=max(candidates,key=lambda c:(c['temporal_development'] is None or c['temporal_development']['passed'],c['metrics']['balanced_accuracy'],-names.index(c['name'])));m=fit(rs,best['name'],field,target);bundle[key]=dict(model=m,key=key,trained_ids=[r['id'] for r in rs],dimensions=len(rs[0][field]),production_replacement=False);reports.append(dict(key=key,status='trained_experimental',train_frames=len(rs),people=len(set(groups)),class_groups=support,selected=best['name'],group_cv=best['metrics'],temporal_development=best['temporal_development'],oof_predictions=best['predictions'],candidates=[dict(name=c['name'],metrics=c['metrics'],temporal_development=c['temporal_development']) for c in candidates]));print('STATE_TRAINED',key,best['metrics'],flush=True)
joblib.dump(dict(schema='reviewed_initial_ready_and_return_state_v3',feature_schema='observed_chain_height_anchor_v3',feature_implementation_sha256=sha(B/'tools/initial-pose-return/recognizer.py'),heads=bundle,manual_initial_reference_required_for_relative_heads=True,labels_sha256=sha(O/'state-labels-v3.frozen.json'),production_replacement=False,grade_model=False),O/'initial-pose-state-candidates.joblib');(O/'state-training.json').write_text(json.dumps(dict(completed=True,heads=reports,Test_used=False,group_cv_not_independent_final_validation=True,calibration_anchor_excluded_from_oof_scoring=True),indent=2));(O/'model-freeze.json').write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_sha256=sha(O/'initial-pose-state-candidates.joblib'),inputs_sha256=sha(O/'state-inputs.frozen.json'),labels_sha256=sha(O/'state-labels-v3.frozen.json'),policy_sha256=sha(POLICY),production_replacement=False),indent=2));print('STATE_COMPLETE',len(bundle),len(data),'held',len(held),flush=True)
