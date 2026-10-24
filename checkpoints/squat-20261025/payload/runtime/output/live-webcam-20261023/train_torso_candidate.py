"""기존 시각 라벨로 몸통 후보와 정상 복귀 허용 범위를 보정한다."""
import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from torso_candidate import torso_features
R=Path(__file__).parent;O=R/'front-candidate'
rows=json.loads((O/'training-rows.json').read_text());annotations={r['id']:r for r in json.loads((R.parents[1]/'output/three-exercise-comparison-20261022/score_annotations.json').read_text())};data=[]
for r in rows:
 z=np.load(r['cache_path']);f=torso_features(z['image_xy']);levels=annotations[r['id']]['assessment']['criterion_levels']
 data.append(dict(id=r['id'],person=r['person'],split=r['split'],features=f.tolist(),return_grade=levels[1],trunk_grade=levels[3]))
tr=[r for r in data if r['split']=='train'];va=[r for r in data if r['split']=='validation'];body=[r for r in tr if r['trunk_grade'] is not None]
x=np.array([r['features'][1:] for r in body]);y=np.array([r['trunk_grade'] for r in body])
ai=ExtraTreesRegressor(n_estimators=100,min_samples_leaf=1,random_state=23,n_jobs=2).fit(x,y)
# 몸통 규칙은 알려진 정상·오류를 이용해 경계를 선택한다. 1등급은 데이터가 없어 중간 공학 구간이다.
options=[]
for j in [1,2,3]:
 vals=sorted(set(r['features'][j] for r in body));cuts=[0.]+[(a+b)/2 for a,b in zip(vals,vals[1:])]+[max(vals)+1]
 for good in cuts:
  for partial in cuts:
   if partial<=good:continue
   pred=[2 if r['features'][j]<=good else 1 if r['features'][j]<=partial else 0 for r in body]
   # 등급 불균형 때문에 오류를 정상으로 덮는 경계를 선택하지 않도록 등급별 평균오차.
   loss=np.mean([np.mean([abs(p-g) for p,g in zip(pred,y) if g==k]) for k in set(y)])
   options.append((float(loss),j,good,partial))
loss,j,good,partial=min(options)
normal_return=[r['features'][0] for r in tr if r['return_grade']==2]
ret_good=max(.08,float(np.percentile(normal_return,95)))
# 정상 자료만으로 실패 등급을 만들지 않는다. 부분/실패 경계는 기존 실험 경계를 보수적으로 유지.
ret_partial=max(.20,ret_good*2)
bundle=dict(version='torso-candidate-v2',body_model=ai,body_rule_feature=j,body_rule_boundaries=[good,partial],return_boundaries=[ret_good,ret_partial])
joblib.dump(bundle,O/'torso-model.joblib')
def summary(rs):
 out=[]
 for r in rs:
  f=r['features'];g=r['trunk_grade'];ap=float(ai.predict([f[1:]])[0]);rp=2 if f[j]<=good else 1 if f[j]<=partial else 0;old=2 if f[1]<=8 else 1 if f[1]<=15 else 0
  out.append(dict(id=r['id'],target=g,old_rule=old,new_rule=rp,ai=ap,return_target=r['return_grade'],return_delta=f[0]))
 known=[r for r in out if r['target'] is not None]
 return dict(rows=out,old_rule_MAE=float(np.mean([abs(r['old_rule']-r['target']) for r in known])),new_rule_MAE=float(np.mean([abs(r['new_rule']-r['target']) for r in known])),ai_MAE=float(np.mean([abs(r['ai']-r['target']) for r in known])))
report=dict(training=summary(tr),validation=summary(va),body_training_grade_counts={str(g):int(sum(y==g)) for g in [0,1,2]},body_error_people=sorted({r['person'] for r in body if r['trunk_grade']<2}),return_training_grade_counts={str(g):sum(r['return_grade']==g for r in tr) for g in [0,1,2]},body_rule_feature=j,body_rule_boundaries=[good,partial],return_boundaries=[ret_good,ret_partial],limitations=['몸통 오류는 학습 사람 1명의 1구간뿐; 오류 일반화 미검증','검증 자료 몸통 오류 0건; 정상 유지 검사만 가능','복귀 실패 라벨 없음: 복귀 AI 분류기를 훈련하지 않음','복귀 허용 경계는 정상 구간으로 보정했으며 오류 등급 경계는 공학 가정','몸통 AI는 진단 후보로만 제공; 합계는 몸통 규칙을 사용'])
(O/'torso-training.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(O/'torso-training-rows.json').write_text(json.dumps(data,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
