"""Group CV selects candidates; fixed development validation gates adoption."""
import sys,collections,time,copy
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from common import B,O,read,save,sha,status

def key(r,i):
 n=r['items'][i];return (r['exercise'],n['item'],n['bank'])
def weights(rs):
 # Equal mass per grade, source family, and connected participant group.
 c=collections.Counter((r['target'],r['family'],r['group']) for r in rs);families=collections.defaultdict(set);groups=collections.defaultdict(set)
 for y,f,g in c:families[y].add(f);groups[y,f].add(g)
 result=np.array([1/(len(families[r['target']])*len(groups[r['target'],r['family']])*c[r['target'],r['family'],r['group']]) for r in rs]);return result*len(result)/result.sum()
def model(name,constraints=None):
 return HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=5,min_samples_leaf=3,l2_regularization=1,random_state=241024,monotonic_cst=constraints if name=='monotone80' else None) if name in ['boost80','monotone80'] else ExtraTreesRegressor(n_estimators=160,min_samples_leaf=int(name[4:]),max_features=1.,n_jobs=2,random_state=241024)
def fit(rs,name):
 constraints=None
 if rs[0]['exercise']=='plank':
  i=rs[0]['index'];dims=len(rs[0]['x'])
  if i==0 and dims==7:constraints=[1]*7
  elif i==1 and dims==7:constraints=[-1]*7
  elif i==2 and dims==22:constraints=[1]*21+[0]
 m=model(name,constraints);m.fit([r['x'] for r in rs],[r['target'] for r in rs],sample_weight=weights(rs));return m
def metrics(ps):
 if not ps:return None
 y=np.array([p['target'] for p in ps]);a=np.array([p['prediction'] for p in ps]);err=y<2;normal=y==2;groups=sorted({p['group'] for p in ps})
 grade_group_losses=[np.mean([np.mean([abs(p['target']-p['prediction']) for p in ps if p['target']==grade and p['group']==group]) for group in {p['group'] for p in ps if p['target']==grade}]) for grade in sorted(set(y))]
 return dict(balanced_group_MAE=float(np.mean(grade_group_losses)),n=len(ps),groups=len(groups),MAE=float(np.mean(abs(y-a))),group_MAE=float(np.mean([np.mean([abs(p['target']-p['prediction']) for p in ps if p['group']==g]) for g in groups])),error_n=int(err.sum()),normal_n=int(normal.sum()),error_recall=float(np.mean(a[err]<1.5)) if err.any() else None,normal_false_alarm=float(np.mean(a[normal]<1.5)) if normal.any() else None)
def predictions(m,rs):
 return [dict(id=r['id'],person=r['person'],group=r['group'],family=r['family'],target=r['target'],prediction=float(p)) for r,p in zip(rs,np.clip(m.predict([r['x'] for r in rs]),0,2))] if rs else []
def main():
 rows=read(O/'score-inputs.frozen.json');buckets=collections.defaultdict(list)
 for r in rows:
  for i,(n,g) in enumerate(zip(r['items'],r['levels'])):
   if g is None or not n['eligible'] or n['feature'] is None:continue
   if not np.isfinite(n['feature']).all():raise ValueError('Nonfinite eligible feature')
   buckets[key(r,i)].append(dict(id=r['id'],exercise=r['exercise'],index=i,x=n['feature'],target=g,person=r['person'],group=r['group'],family=r['family'],origin=r['origin'],split=r['split']))
 reports=[];heads={};rowpreds=collections.defaultdict(dict)
 for k,rs in sorted(buckets.items()):
  started=time.perf_counter()
  assert len({len(r['x']) for r in rs})==1,k
  tr=[r for r in rs if r['split']=='train'];va=[r for r in rs if r['split']=='validation'];assert not {r['group'] for r in tr}&{r['group'] for r in va};support=collections.Counter(r['target'] for r in tr);normal={r['group'] for r in tr if r['target']==2};errors={r['group'] for r in tr if r['target']<2};report=dict(key=list(k),train_count=len(tr),validation_count=len(va),grade_counts=dict(support),normal_groups=len(normal),error_groups=len(errors),source_families=dict(collections.Counter(r['family'] for r in tr)),feature_dimensions=len(rs[0]['x']))
  if len(support)<2 or min(len(normal),len(errors))<2:
   report.update(status='unsupported_missing_normal_or_error_groups',runtime_replacement=False);reports.append(report);save('score-training.json',dict(heads=reports,completed=False));continue
  cv=[];group=np.array([r['group'] for r in tr]);splits=list(GroupKFold(n_splits=min(4,len(set(group)))).split(tr,groups=group))
  for name in ['tree2','tree4','tree8','boost80']+(['monotone80'] if k[0]=='plank' and k[1] in ['trunk_line','elbow_sagittal_position','leg_and_body_support'] else []):
   ps=[];base=[]
   for ti,vi in splits:
    t=[tr[i] for i in ti];v=[tr[i] for i in vi];m=fit(t,name);ps+=predictions(m,v);mean=float(np.average([r['target'] for r in t],weights=weights(t)));base.extend([dict(id=r['id'],group=r['group'],target=r['target'],prediction=mean) for r in v])
   cv.append(dict(name=name,metrics=metrics(ps),constant_baseline=metrics(base),predictions=ps))
  best=min(cv,key=lambda c:c['metrics']['balanced_group_MAE']);m=fit(tr,best['name']);ps=predictions(m,va);actual=None
  path=B/'team-posture-scoring/Server/AIService/artifacts/posture_score'/(k[0]+'.joblib');old=joblib.load(path)
  h=old.get(k)
  if k[0]=='ytw':
   legacyids=['y_reach','t_reach','w_shape','hinge_and_trunk_hold'];h=old.get(legacyids.index(k[1])) if k[1] in legacyids and k[2]!='upright_temporal_v1' else None
  if k[1]=='standing_height_return':h=joblib.load(B/'output/squat-return-20261023/candidate.joblib')
  if h and not isinstance(h.get('model'),dict) and h['model'].n_features_in_==len(rs[0]['x']):actual=predictions(h['model'],va)
  oldmetrics=metrics(actual) if actual else None;vm=metrics(ps);cvmet=best['metrics'];eligible=cvmet['balanced_group_MAE']<best['constant_baseline']['balanced_group_MAE'] and cvmet['error_recall'] is not None and cvmet['error_recall']>=.65 and cvmet['normal_false_alarm'] is not None and cvmet['normal_false_alarm']<=.25
  improved=eligible and (oldmetrics is None or (vm['MAE']<=oldmetrics['MAE'] and (oldmetrics['error_recall'] is None or vm['error_recall'] is not None and vm['error_recall']>=oldmetrics['error_recall']) and (oldmetrics['normal_false_alarm'] is None or vm['normal_false_alarm'] is not None and vm['normal_false_alarm']<=oldmetrics['normal_false_alarm'])))
  head=dict(model=m,key=k,feature_dimensions=len(rs[0]['x']),grade_support=dict(support),trained_ids=[r['id'] for r in tr],candidate_gate_passed=bool(improved),production_replacement=False);heads[k]=head
  for p in ps:rowpreds[p['id']][k[1]]=p['prediction']
  report.update(fit_and_cv_seconds=time.perf_counter()-started,status='trained_experimental',selected=best['name'],group_cv=cvmet,constant_group_cv=best['constant_baseline'],fixed_development_validation=vm,current_saved_model_same_input=oldmetrics,candidate_gate_passed=bool(improved),runtime_replacement=False,validation_predictions=ps,current_model_predictions=actual,candidate_cv=[{a:b for a,b in c.items() if a!='predictions'} for c in cv],selected_oof_predictions=best['predictions']);reports.append(report);save('score-training.json',dict(heads=reports,completed=False));status('score_head_trained',head=list(k),train=len(tr),gate=bool(improved))
 joblib.dump(dict(schema='integrated-five-criterion-candidates-v1',heads=heads,labels_sha256=sha(O/'score-inputs.frozen.json'),review_status='non_expert_visual_draft',runtime_replacement=False),O/'score-candidates.joblib');save('score-training.json',dict(heads=reports,completed=True,trained_heads=len(heads),gate_passed=sum(r.get('candidate_gate_passed',False) for r in reports),final_test_used=False,validation_is_development=True,models_promoted_to_live_service=False,missing_heads_not_filled_by_constant_normal=True));save('validation-item-predictions.json',rowpreds)
 status('score_training_complete',trained=len(heads))
if __name__=='__main__':main()
