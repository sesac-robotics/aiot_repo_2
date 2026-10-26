from pathlib import Path
import json,collections
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
O=Path(__file__).resolve().parent;r=json.loads((O/'feature-rows.json').read_text());tr=[r for r in r if r['split']=='train'];te=[r for r in r if r['split']=='validation']
def metric(ps):return dict(miss=sum(y==1 and p>=1.5 for y,p in ps),false=sum(y==2 and p<1.5 for y,p in ps),mae=float(np.mean([abs(y-p) for y,p in ps])))
for name,ctor in [('logistic',lambda:make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced'))),('ridge',lambda:make_pipeline(StandardScaler(),Ridge(alpha=10))),('svm',lambda:make_pipeline(StandardScaler(),SVC(C=1,class_weight='balanced',probability=True,random_state=23))),('tree',lambda:DecisionTreeRegressor(max_depth=3,min_samples_leaf=1,random_state=23)),('forest',lambda:RandomForestRegressor(n_estimators=120,min_samples_leaf=1,random_state=23)),('extra',lambda:ExtraTreesRegressor(n_estimators=160,min_samples_leaf=1,random_state=23))]:
 def fit(rs):
  m=ctor();cnt=collections.Counter(r['target'] for r in rs);x=[r['features']['x'] for r in rs];y=[r['target'] for r in rs]
  if name in ['tree','forest','extra']:m.fit(x,y,sample_weight=[1/cnt[y] for y in y])
  else:m.fit(x,y)
  return m
 def predict(m,rs):
  x=[r['features']['x'] for r in rs]
  if name in ['logistic','svm']:p=1+m.predict_proba(x)[:,1]
  else:p=m.predict(x)
  return list(zip([r['target'] for r in rs],p))
 ps=[]
 for p in sorted({r['person'] for r in tr}):ps+=predict(fit([r for r in tr if r['person']!=p]),[r for r in tr if r['person']==p])
 print(name,'cv',metric(ps),'held',metric(predict(fit(tr),te)))
