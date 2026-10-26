from pathlib import Path
import json,collections,hashlib
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesRegressor
from features import extract,reference,NAMES,VERSION
NAMES=NAMES+['왼쪽 골반-무릎 높이 복귀 차','왼쪽 허벅지 투영 복귀 차','오른쪽 골반-무릎 높이 복귀 차','오른쪽 허벅지 투영 복귀 차']
O=Path(__file__).resolve().parent;labels=json.loads((O/'labels.frozen.json').read_text())['labels'];rows=[];excluded=[]
for r in labels:
 if r['grade'] is None:continue
 name=Path(r['path']).stem;z=np.load(O/(name+'-heavy.npz'));frames=z['frame_numbers'];xy=z['image_xy'];first=18 if name=='squat_diag_m4_5_7' else min(x['last_frame'] for x in labels if Path(x['path']).stem==name and x['grade']==2)-6
 if r['first_frame']<first+6:excluded.append(r['id']);continue
 ref=reference(xy[(frames>=first)&(frames<=first+8)]);mask=(frames>=r['first_frame'])&(frames<=r['last_frame']);f=extract(xy[mask],ref,int(z['width']),int(z['height']))
 if f and f['optional_knee_context'] is not None:
  f['x']=f['x']+f['optional_knee_context'];rows.append(dict(id=r['id'],person=r['person'],source=name,view='front' if '_front_' in name else 'diagonal',split=r['split'],target=r['grade'],features=f))
 else:excluded.append(r['id'])
(O/'feature-rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
def fit(rs):
 counts=collections.Counter(r['target'] for r in rs)
 return ExtraTreesRegressor(n_estimators=160,min_samples_leaf=2,max_features=1.,random_state=23,n_jobs=2).fit([r['features']['x'] for r in rs],[r['target'] for r in rs],sample_weight=[1/counts[r['target']] for r in rs])
def predict(model,rs):
 return [dict(id=r['id'],person=r['person'],view=r['view'],target=r['target'],prediction=float(p)) for r,p in zip(rs,model.predict([r['features']['x'] for r in rs]))] if rs else []
def metrics(ps):
 return dict(n=len(ps),partial_n=sum(p['target']==1 for p in ps),normal_n=sum(p['target']==2 for p in ps),missed_partial=sum(p['target']==1 and p['prediction']>=1.5 for p in ps),normal_flagged=sum(p['target']==2 and p['prediction']<1.5 for p in ps),MAE_points=float(np.mean([abs(p['prediction']-p['target'])*10 for p in ps])) if ps else None)
train=[r for r in rows if r['split']=='train'];held=[r for r in rows if r['split']=='validation'];folds=[];oof=[]
for person in sorted({r['person'] for r in train}):
 tr=[r for r in train if r['person']!=person];te=[r for r in train if r['person']==person];assert not {r['source'] for r in tr}&{r['source'] for r in te};ps=predict(fit(tr),te);folds.append(dict(held_person=person,metrics=metrics(ps)));oof+=ps
model=fit(train);ps=predict(model,held);report=dict(feature_version=VERSION,train_support=dict(collections.Counter(str(r['target']) for r in train)),train_people=sorted({r['person'] for r in train}),validation_people=sorted({r['person'] for r in held}),folds=folds,group_cv=metrics(oof),fixed_validation=metrics(ps),predictions=oof+ps,excluded=excluded,grade_zero_supported=False,independent_final_test=False,expert_validated=False)
report['development_gate_passed']= bool(report['group_cv']['missed_partial']==0 and report['group_cv']['normal_flagged']<=2 and report['fixed_validation']['missed_partial']==0 and report['fixed_validation']['normal_flagged']<=1)
labelsha=hashlib.sha256((O/'labels.frozen.json').read_bytes()).hexdigest();bundle=dict(model=model,feature_version=VERSION,feature_names=NAMES,train_support=report['train_support'],training_people=report['train_people'],labels_sha256=labelsha,runtime_approved=report['development_gate_passed'],supported_grades=[1,2],limitations=['0등급 심한 미복귀 자료 미확보','전문가 정답 검증 아님','정면·사선 개발 검증, 독립 최종시험 아님'])
joblib.dump(bundle,O/'candidate.joblib');report['model_sha256']=hashlib.sha256((O/'candidate.joblib').read_bytes()).hexdigest();(O/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['predictions','excluded']},ensure_ascii=False,indent=2))
