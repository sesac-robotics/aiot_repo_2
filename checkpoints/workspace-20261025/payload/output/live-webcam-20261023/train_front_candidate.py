"""기존 train만 학습하고 기존 validation·로컬 5회는 학습하지 않는다."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from front_candidate import vector,rule_depth,predict
from runtime_snapshot import read_source,digest
ROOT=Path(__file__).parent;OUT=ROOT/'front-candidate';OUT.mkdir(exist_ok=True)
BASE=ROOT.parents[1];OLD=BASE/'output/three-exercise-comparison-20261022'
annotations=json.loads((OLD/'score_annotations.json').read_text());cache={r['id']:r for r in json.loads((OLD/'cache_manifest.json').read_text())['records']}
source_before=digest(read_source());rows=[];rejected=[]
for r in annotations:
 if r.get('exercise')!='squats' or r['split'] not in ('train','validation'):continue
 entry=cache.get(r['id'],{}).get('paths',{}).get('front')
 if not entry:continue
 level=r.get('assessment',{}).get('criterion_levels',[None])[0]
 if level is None:continue
 try:
  z=np.load(entry['path']);v=vector(z['image_xy'])
 except ValueError as e:rejected.append(dict(id=r['id'],reason=str(e)));continue
 rows.append(dict(id=r['id'],person=r['person'],split=r['split'],target=level,x=v.tolist(),cache_path=entry['path'],label_source=r['label_source'],source_sha256=r.get('video_sha256'),front_source=r.get('front_path')))
train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='validation']
assert not {r['person'] for r in train}&{r['person'] for r in val}
x=np.array([r['x'] for r in train]);y=np.array([r['target'] for r in train])
assert set(y)=={0,1,2}
model=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=2,max_features=1.,random_state=23,n_jobs=2).fit(x,y)
# 규칙은 방향이 명확한 골반 세로 이동량 특징만 사용하며 경계는 train에서만 결정한다.
feature=0;values=sorted(set(x[:,feature]));cuts=[0.]+[(a+b)/2 for a,b in zip(values,values[1:])]+[max(values)+.01]
options=[]
for lo in cuts:
 for hi in cuts:
  if lo>=hi:continue
  error=float(np.mean([abs(rule_depth(v,(lo,hi))-g) for v,g in zip(x[:,feature],y)]))
  options.append((error,lo,hi))
_,lo,hi=min(options)
bundle=dict(version='front-depth-estimator-v1',model=model,rule_feature=feature,rule_boundaries=[lo,hi],label_scope='측면 시각 깊이 초안을 동기화 정면 입력으로 추정; 전문가 정답 아님',trained_ids=[r['id'] for r in train])
joblib.dump(bundle,OUT/'model.joblib')
def metrics(rs):
 xx=np.array([r['x'] for r in rs]);yy=np.array([r['target'] for r in rs]);ai=model.predict(xx);rule=np.array([rule_depth(v[feature],(lo,hi)) for v in xx])
 return dict(n=len(rs),people=sorted({r['person'] for r in rs}),ai_grade_MAE=float(np.mean(abs(ai-yy))),rule_grade_MAE=float(np.mean(abs(rule-yy))),grade_counts={str(g):int(sum(yy==g)) for g in [0,1,2]},predictions=[dict(id=r['id'],target=r['target'],ai=float(a),rule=int(b)) for r,a,b in zip(rs,ai,rule)])
p=ROOT/'recordings/20261023-110200-964840';z=np.load(p/'pose.npz');labels=json.loads((p/'labels.frozen.json').read_text());local=[]
for r in labels['repetitions']:
 local.append(dict(number=r['number'],**predict(z['image_xy'][r['start_frame']:r['end_frame']+1])))
report=dict(training=metrics(train),reused_development_validation=metrics(val),rule_boundaries=[lo,hi],rejected=rejected,local_not_used_in_training=True,local_depth_ground_truth=None,source_preserved=source_before==digest(read_source()),model_sha256=hashlib.sha256((OUT/'model.joblib').read_bytes()).hexdigest(),candidate_only=True,independent_final_test=False,local=local)
(OUT/'training-rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False))
