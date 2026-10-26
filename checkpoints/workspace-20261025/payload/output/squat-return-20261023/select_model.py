from pathlib import Path
import json,collections
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
O=Path(__file__).resolve().parent;rows=json.loads((O/'feature-rows.json').read_text());tr=[r for r in rows if r['split']=='train'];results=[]
for idx in [[0],[3],[0,3],[0,1,2],[0,1,2,3],[0,1,2,3,4,5]]:
 for name in ['logistic','extra','hist']:
  ps=[]
  for person in sorted({r['person'] for r in tr}):
   train=[r for r in tr if r['person']!=person];test=[r for r in tr if r['person']==person];cnt=collections.Counter(r['target'] for r in train);x=[[r['features']['x'][i] for i in idx] for r in train];y=[r['target'] for r in train];xx=[[r['features']['x'][i] for i in idx] for r in test]
   if name=='logistic':m=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced'));m.fit(x,y);p=1+m.predict_proba(xx)[:,1]
   else:
    m=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=2,random_state=23,n_jobs=2) if name=='extra' else HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=3,min_samples_leaf=2,l2_regularization=1,monotonic_cst=[-1 if i!=4 else 1 for i in idx],random_state=23)
    m.fit(x,y,sample_weight=[1/cnt[v] for v in y]);p=m.predict(xx)
   ps += [dict(id=r['id'],target=r['target'],prediction=float(v)) for r,v in zip(test,p)]
  err=[p for p in ps if p['target']==1];normal=[p for p in ps if p['target']==2];d=dict(name=name,features=idx,miss=sum(p['prediction']>=1.5 for p in err),false=sum(p['prediction']<1.5 for p in normal),balanced_MAE=float(np.mean([np.mean([abs(p['prediction']-p['target']) for p in z]) for z in [err,normal]])),predictions=ps);results.append(d);print(name,idx,d['miss'],d['false'],round(d['balanced_MAE'],3),flush=True)
(O/'model-selection.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
