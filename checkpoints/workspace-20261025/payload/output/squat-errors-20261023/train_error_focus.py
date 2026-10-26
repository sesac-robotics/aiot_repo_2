"""Nested person-group development experiment; no new normal labels or pseudo labels."""
from pathlib import Path
import json,collections,hashlib
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
O=Path(__file__).resolve().parent;P=O.parent/'squat-priority-20261023';prior=json.loads((P/'comparison.json').read_text());rows=prior['feature_rows'];reports=[];heads={}
def fit(rows,key,mult=1):
 y=[r['levels'][key] for r in rows];c=collections.Counter(y);m=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=2,max_features=1.,random_state=23,n_jobs=2)
 return m.fit([r['new_features'][key]['x'] for r in rows],y,sample_weight=[(mult if g<2 else 1)/c[g] for g in y])
def metrics(ps):
 y=np.array([x['target'] for x in ps]);v=np.array([x['pred'] for x in ps]);err=y<2;normal=y==2
 return dict(n=len(y),error_n=int(err.sum()),normal_n=int(normal.sum()),missed_errors=int(np.sum((v>=1.5)&err)),normal_flagged=int(np.sum((v<1.5)&normal)),MAE=float(np.mean(abs(v-y))),error_MAE=float(np.mean(abs(v[err]-y[err]))) if any(err) else None,balanced_MAE=float(np.mean([np.mean(abs(v[y==g]-g)) for g in sorted(set(y))])))
def choose(train,key):
 groups=[x['person'] for x in train];scores=[]
 for mult in [1,2,4]:
  ps=[]
  for tr,te in GroupKFold(min(3,len(set(groups)))).split(train,groups=groups):
   model=fit([train[i] for i in tr],key,mult)
   for i in te:ps.append(dict(target=train[i]['levels'][key],pred=float(model.predict([train[i]['new_features'][key]['x']])[0])))
  m=metrics(ps);score=m['balanced_MAE']+ .5*m['missed_errors']/max(1,m['error_n'])+.25*m['normal_flagged']/max(1,m['normal_n']);scores.append(dict(mult=mult,score=score,metrics=m))
 return min(scores,key=lambda x:(x['score'],x['mult']))['mult'],scores
for h in prior['heads']:
 key=h['item'];cases=[r for r in rows if r['levels'].get(key) is not None and key in r['new_features'] and key in r['features']];folds=[];before=[];after=[]
 for fold,f in enumerate(h['folds']):
  train=[r for r in cases if r['person'] in f['train_people']];held=[r for r in cases if r['id'] in f['held_ids']];assert not {r['person'] for r in train}&{r['person'] for r in held};assert not {s for r in train for s in r['source']}&{s for r in held for s in r['source']}
  mult,scores=choose(train,key);bm=fit(train,key);am=fit(train,key,mult)
  for r in held:
   base=dict(id=r['id'],person=r['person'],target=r['levels'][key],fold=fold);before.append(dict(base,pred=float(bm.predict([r['new_features'][key]['x']])[0])));after.append(dict(base,pred=float(am.predict([r['new_features'][key]['x']])[0])))
  folds.append(dict(fold=fold,train_people=f['train_people'],held_people=f['held_people'],selected_error_weight=mult,inner_results=scores))
 b=metrics(before);a=metrics(after);eligible=a['missed_errors']<b['missed_errors'] and a['normal_flagged']<=b['normal_flagged'] and a['error_MAE']<=b['error_MAE'] and a['balanced_MAE']<=b['balanced_MAE']
 train=[r for r in cases if r['split']=='train'];mult,scores=choose(train,key);heads[key]=dict(model=fit(train,key,mult),feature_version='geometry-context-v1',selected_error_weight=mult,train_ids=[r['id'] for r in train],runtime_approved=False,development_improved=eligible)
 reports.append(dict(item=key,before=b,after=a,improved=eligible,folds=folds,before_predictions=before,after_predictions=after,final_train_inner=scores));print(key,b,a,eligible,flush=True)
joblib.dump(dict(heads=heads,version='error-cost-nested-development-v1',source_digest=prior['source_digest'],label_sha256=prior['labels_sha256'],additional_normal_rows=0,additional_error_rows=0),O/'error-focus-candidate.joblib')
(O/'error-focus-comparison.json').write_text(json.dumps(dict(reports=reports,criterion_unchanged=True,new_normal_training_rows=0,new_error_training_rows=0,baseline='Previous geometry-context candidate recipe refit in identical outer folds',threshold_note='1.5 is the existing internal diagnostic boundary, not an athlete score or clinical cutoff',independent_final_test=False,runtime_models_changed=False,limitations=['기존 개발 자료를 재사용한 추가 실험. 독립 최종 성능이 아님','New source review labels are not included in this experiment']),ensure_ascii=False,indent=2))
