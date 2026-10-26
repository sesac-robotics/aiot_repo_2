"""Original trunk feature/weight alignment; existing visual labels, isolated artifacts."""
import sys,json,hashlib
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesRegressor
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent;L=R/'output/live-webcam-20261023';S=R/'output/squat-relabel-expanded-20261023';sys.path.insert(0,str(L))
from runtime_snapshot import prepare_snapshot,read_source,digest
runtime,source_sha=prepare_snapshot();sys.path.insert(0,str(runtime))
from app.scoring import three_items
raw=(S/'labels.frozen.json').read_bytes();labels=json.loads(raw)['records'];out=[]
# Map original trunk semantics only. Do not substitute knee gap for composite alignment,
# hip-height return for joint extension, or frontal drop for measured depth.
for r in labels:
 z=np.load(S/'observations'/r['id']/'front.npz');x=three_items.mask_image_border(z['image_xy'],float(z['width']),float(z['height']));note=three_items.describe('squats',None,x)[3]
 out.append(dict(id=r['id'],person=r['person'],split=r['split'],sources=r['sources'],target=r['levels']['trunk_balance'],feature=note['feature'] if note['eligible'] else None,measurement=note['measurement'],source_evidence=r['evidence']['trunk_balance'],review_evidence=r['review_evidence'],action='reuse_same_semantics',expert_validated=False))
# Targeted visual re-review of 4 original borderline/error montage rows; no prediction used for labels.
review={'Petra_03':'하단 여러 표본에서 명확한 좌우 기울임; 기존 0 유지','Terka_03':'좌우 기울임이 작고 부분적; 기존 1 유지','Vasil_03':'하단 여러 표본에서 명확한 좌우 기울임; 기존 0 유지','Mima_03':'좌우 기울임이 부분적; 기존 1 유지'}
for r in out:
 if r['id'] in review:r.update(action='targeted_visual_rereview_same_label',review_note=review[r['id']],review_frames=9,continuous_video_review=False)
label_doc=dict(version='original100-trunk-label-alignment-v1',original_item='trunk_balance',weight=20,label_source_sha256=hashlib.sha256(raw).hexdigest(),labels_changed=0,rows=out,expert_validated=False,frozen_before_training=True)
(O/'labels.aligned.frozen.json').write_text(json.dumps(label_doc,ensure_ascii=False,indent=2));label_sha=hashlib.sha256((O/'labels.aligned.frozen.json').read_bytes()).hexdigest()
tr=[r for r in out if r['split']=='train' and r['feature'] is not None and r['target'] is not None];va=[r for r in out if r['split']=='validation' and r['feature'] is not None and r['target'] is not None]
assert not {r['person'] for r in tr}&{r['person'] for r in va};assert not {s for r in tr for s in r['sources']}&{s for r in va for s in r['sources']}
y=np.array([r['target'] for r in tr]);support={g:int(sum(y==g)) for g in (0,1,2)}
model=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=2,random_state=23,n_jobs=2).fit([r['feature'] for r in tr],y,sample_weight=[1/support[g] for g in y])
original=joblib.load(runtime/'artifacts/posture_score/squats.joblib');head=original[('squats','trunk_balance','front')]
preds=[]
for r in va:
 before=float(head['model'].predict([r['feature']])[0]);after=float(model.predict([r['feature']])[0]);v=r['measurement'];good,partial=head['boundary'];rule=2 if v<=good else 1 if v<=partial else 0
 preds.append(dict(id=r['id'],target=r['target'],before_raw_ai=before,after_ai=after,before_rule=rule,after_rule=rule))
mae={k:float(np.mean([abs(r[k]-r['target']) for r in preds])) for k in ['before_raw_ai','after_ai','before_rule','after_rule']}
# A raw constant-normal baseline is not a validated AI. Still do not replace it
# claiming better accuracy when the observed error grows.
report=dict(version='original100-alignment-v1',source_digest=source_sha,label_sha256=label_sha,train_count=len(tr),validation_count=len(va),train_support=support,train_people=sorted({r['person'] for r in tr}),validation_people=sorted({r['person'] for r in va}),validation_MAE_grade=mae,validation_MAE_points20={k:v*10 for k,v in mae.items()},predictions=preds,promoted=bool(mae['after_ai']<=mae['before_raw_ai']),selection_warning='개발 검증 확인에 따른 보수적 반영 판단; 독립 최종 시험 아님',test_opened=False,expert_validated=False,rule_boundaries=list(head['boundary']),feature_pipeline='original three_items.describe: abs frontal trunk tilt percentiles5/10/25/50/75/90/95+coverage',original_single_grade_model=True)
joblib.dump(dict(model=model,version='original100-trunk-candidate-v1',label_sha256=label_sha,source_digest=source_sha,train_support=support,rule_boundaries=list(head['boundary'])),O/'trunk-candidate.joblib')
report['model_sha256']=hashlib.sha256((O/'trunk-candidate.joblib').read_bytes()).hexdigest();assert source_sha==digest(read_source());(O/'training-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
