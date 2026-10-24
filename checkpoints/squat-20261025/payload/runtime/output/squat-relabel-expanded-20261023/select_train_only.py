"""Select configurations using training people only, never validation targets."""
from pathlib import Path
import json,joblib,numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
P=Path(__file__).parent;bundle=joblib.load(P/'candidate.joblib');rows=json.loads((P/'feature-rows.json').read_text());summary={}
for key,h in bundle['heads'].items():
 tr=[r for r in rows if r['split']=='train' and r['levels'][key] is not None and key in r['features']]
 people=sorted({r['person'] for r in tr});n=len(tr[0]['features'][key]['x']);candidates=[('all_leaf2',list(range(n)),2,'forest'),('relative_leaf1',list(range(3 if key=='knee_inward' else n)),1,'forest'),('scalar_tree', [0 if key=='knee_inward' else 2],1,'tree')];metrics=[]
 def fit(samples,indices,leaf,kind):
  x=np.array([[r['features'][key]['x'][j] for j in indices] for r in samples]);y=np.array([r['levels'][key] for r in samples]);counts={g:sum(y==g) for g in set(y)};w=[1/counts[g] for g in y]
  model=DecisionTreeRegressor(max_depth=2,min_samples_leaf=leaf,random_state=23) if kind=='tree' else ExtraTreesRegressor(n_estimators=120,min_samples_leaf=leaf,random_state=23,n_jobs=2)
  return model.fit(x,y,sample_weight=w)
 for name,indices,leaf,kind in candidates:
  errors=[]
  for person in people:
   training=[r for r in tr if r['person']!=person];held=[r for r in tr if r['person']==person]
   if len({r['levels'][key] for r in training})<2:continue
   model=fit(training,indices,leaf,kind)
   for r in held:
    prediction=float(model.predict([[r['features'][key]['x'][j] for j in indices]])[0]);errors.append((r['levels'][key],abs(prediction-r['levels'][key])))
  score=float(np.mean([np.mean([e for g,e in errors if g==grade]) for grade in {g for g,e in errors}])) if errors else float('inf')
  metrics.append(dict(name=name,indices=indices,leaf=leaf,kind=kind,balanced_person_holdout_MAE=score,heldout_cases=len(errors)))
 best=min(metrics,key=lambda a:a['balanced_person_holdout_MAE']);h['model']=fit(tr,best['indices'],best['leaf'],best['kind']);h['feature_indices']=best['indices'];summary[key]=dict(selection=best,candidates=metrics,validation_used_for_selection=False)
bundle['selection']='train-person-holdout-only';joblib.dump(bundle,P/'selected-candidate.joblib');(P/'selection.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,indent=2))
