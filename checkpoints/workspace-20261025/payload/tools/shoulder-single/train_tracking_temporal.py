"""Second exposed-development candidate adds causal four-joint continuity, no posture tilt."""
from pathlib import Path
import os,json,sys
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesClassifier
from tracking_quality import features
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-single-audit-v1-20261025';labels=json.loads((O/'tracking-labels.frozen.json').read_text());audit={r['id']:r for r in json.loads((O/'train-provenance-audit.json').read_text())['rows']};rows=[]
for r in labels:
 z=np.load(r['cache']);i=r['cache_sample_index'];fps=audit[r['source_id']]['source_fps'];times=z['frame_numbers']/fps;now=features(z['image_xy'][i]);previous=[]
 for j in range(max(0,i-5),i):
  if times[i]-times[j]>.61:continue
  f=features(z['image_xy'][j])
  if f is not None:previous.append((j,f))
 # Missing history is an explicit flag; no imputed joints or made-up feature observations.
 if not previous:
  continue
 last,prev=previous[-1];delta=abs(now-prev);hist=np.array([p[1] for p in previous]);features_v2=np.r_[now,delta,np.max(abs(hist-now),axis=0),len(previous)/5, times[i]-times[last]]
 rows.append(dict(r,feature=features_v2.tolist()))
def make():return ExtraTreesClassifier(n_estimators=192,min_samples_leaf=2,max_features=None,class_weight='balanced',random_state=20261025,n_jobs=1)
cv=[]
for person in sorted({r['person'] for r in rows}):
 tr=[r for r in rows if r['person']!=person];te=[r for r in rows if r['person']==person];m=make().fit([r['feature'] for r in tr],[r['tracking_error'] for r in tr]);ps=m.predict_proba([r['feature'] for r in te])[:,list(m.classes_).index(1)]
 cv += [dict(id=r['id'],person=person,label=r['tracking_error'],baseline=r['baseline_quality_flag'],candidate=bool(p>=.5),probability=float(p)) for r,p in zip(te,ps)]
def metric(key):return {'errors':sum(r['label']==1 for r in cv),'detected':sum(r['label']==1 and r[key] for r in cv),'accepted':sum(r['label']==0 for r in cv),'false_flags':sum(r['label']==0 and r[key] for r in cv)}
m=make().fit([r['feature'] for r in rows],[r['tracking_error'] for r in rows]);joblib.dump({'schema':'shoulder-causal-four-joint-tracking-quality-v2','model':m,'grade_prediction_enabled':False,'score_prediction_enabled':False,'production_eligible':False},O/'tracking-quality-temporal-v2.joblib');report={'rows':len(rows),'baseline':metric('baseline'),'candidate':metric('candidate'),'cv':cv,'trained_on':'Train only; people excluded as groups','selection':'v2 designed after seeing v1 failures; all exposed development','future_samples_used':False,'score_enabled':False};(O/'tracking-temporal-v2-metrics.json').write_text(json.dumps(report,indent=2)+'\n');(O/'tracking-temporal-v2-inputs.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='cv'}))
