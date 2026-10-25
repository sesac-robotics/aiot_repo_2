"""Exposed development v3: regularized shape-only tracking classifier, no posture scoring."""
from pathlib import Path
import os,json
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
O=Path('/home/jayden/SeSACPAI-Mid/output/shoulder-single-audit-v1-20261025');rows=json.loads((O/'tracking-labels.frozen.json').read_text())
def make():return make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',max_iter=1000,random_state=20261025))
cv=[]
for person in sorted({r['person'] for r in rows}):
 tr=[r for r in rows if r['person']!=person];te=[r for r in rows if r['person']==person];m=make().fit([r['feature'] for r in tr],[r['tracking_error'] for r in tr]);ps=m.predict_proba([r['feature'] for r in te])[:,1]
 cv += [dict(id=r['id'],person=person,label=r['tracking_error'],baseline=r['baseline_quality_flag'],candidate=bool(p>=.5),probability=float(p)) for r,p in zip(te,ps)]
def metric(key):return {'errors':sum(r['label']==1 for r in cv),'detected':sum(r['label']==1 and r[key] for r in cv),'accepted':sum(r['label']==0 for r in cv),'false_flags':sum(r['label']==0 and r[key] for r in cv)}
m=make().fit([r['feature'] for r in rows],[r['tracking_error'] for r in rows]);joblib.dump({'schema':'shoulder-four-joint-tracking-shape-v1','model':m,'target':'tracking_error_not_posture_grade','grade_prediction_enabled':False,'score_prediction_enabled':False,'production_eligible':False},O/'tracking-quality-linear-v3.joblib');report={'rows':len(rows),'baseline':metric('baseline'),'candidate':metric('candidate'),'cv':cv,'selection':'C1 class-balanced fixed threshold0.5; designed after v1/v2 exposed development failures','score_enabled':False,'production_changed':False};(O/'tracking-linear-v3-metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='cv'}))
