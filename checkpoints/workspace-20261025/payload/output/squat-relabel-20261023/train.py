"""고정된 새 라벨의 train만 AI·규칙 학습에 사용하고 validation은 평가만 한다."""
import json,hashlib,sys
from pathlib import Path
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesRegressor
from criteria_features import observed_features
O=Path(__file__).parent;B=O.parents[1];L=B/'output/live-webcam-20261023';sys.path.insert(0,str(L));from runtime_snapshot import read_source,digest
before=digest(read_source());raw=(O/'labels.frozen.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==(O/'labels.sha256').read_text();labels=json.loads(raw)['records'];rows=[]
for r in labels:
 assert r['split'] in ('train','validation');z=np.load(O/'observations'/r['id']/'front.npz');f=observed_features(z['image_xy']);rows.append(dict(**r,features=f))
train_people={r['person'] for r in rows if r['split']=='train'};val_people={r['person'] for r in rows if r['split']=='validation'};assert not train_people&val_people
assert not {s for r in rows if r['split']=='train' for s in r['sources']} & {s for r in rows if r['split']=='validation' for s in r['sources']}
keys=['knee_inward','descent_range','standing_return','trunk_balance','heel_support','spine_control'];models={};reports={}
def rule(v,bounds,higher):
 good,partial=bounds
 return 2 if (v>=good if higher else v<=good) else 1 if (v>=partial if higher else v<=partial) else 0
for key in keys:
 tr=[r for r in rows if r['split']=='train' and r['levels'][key] is not None and key in r['features']];va=[r for r in rows if r['split']=='validation' and r['levels'][key] is not None and key in r['features']]
 support={str(g):len([r for r in tr if r['levels'][key]==g]) for g in [0,1,2]};error_people={r['person'] for r in tr if r['levels'][key]<2};good_people={r['person'] for r in tr if r['levels'][key]==2}
 if len(error_people)<2 or len(good_people)<2:
  reports[key]=dict(trained=False,train_support=support,reason='관측된 정상·오류 각각 2명 이상 필요',observed_train=len(tr),observed_validation=len(va));continue
 x=np.array([r['features'][key]['x'] for r in tr]);y=np.array([r['levels'][key] for r in tr]);w=np.array([1/support[str(int(g))] for g in y])
 model=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=2,max_features=1.,random_state=23,n_jobs=2).fit(x,y,sample_weight=w)
 vals=sorted(set(r['features'][key]['scalar'] for r in tr));cuts=[vals[0]-.01]+[(a+b)/2 for a,b in zip(vals,vals[1:])]+[vals[-1]+.01];higher=tr[0]['features'][key]['higher'];options=[]
 for good in cuts:
  for partial in cuts:
   if (higher and good<=partial) or (not higher and good>=partial):continue
   pred=[rule(r['features'][key]['scalar'],(good,partial),higher) for r in tr];loss=float(np.average(abs(np.array(pred)-y),weights=w));options.append((loss,good,partial))
 _,good,partial=min(options);models[key]=dict(model=model,boundaries=[good,partial],higher=higher,train_support=support,training_people=sorted(train_people),error_people=sorted(error_people))
 preds=[]
 for r in va:
  f=r['features'][key];ai=float(np.clip(model.predict([f['x']])[0],0,2));rp=rule(f['scalar'],(good,partial),higher);target=r['levels'][key]
  # 비교 가능한 기존 관측 규칙만 비교한다. 깊이·무릎발의 옛 항목을 새 항목으로 강제 치환하지 않는다.
  baseline=(2 if f['scalar']>=.75 else 0) if key=='knee_inward' else (2 if f['scalar']<=8 else 1 if f['scalar']<=15 else 0) if key=='trunk_balance' else None
  preds.append(dict(id=r['id'],target=target,ai=ai,rule=rp,baseline_observation_rule=baseline))
 reports[key]=dict(trained=True,train_support=support,observed_train=len(tr),validation_cases=len(preds),error_people=sorted(error_people),boundaries=[good,partial],ai_grade_MAE=float(np.mean([abs(r['ai']-r['target']) for r in preds])) if preds else None,rule_grade_MAE=float(np.mean([abs(r['rule']-r['target']) for r in preds])) if preds else None,predictions=preds)
joblib.dump(dict(schema='squat-checklist-heads-v1',heads=models,label_sha256=hashlib.sha256(raw).hexdigest(),no_total_weights=True),O/'candidate.joblib')
report=dict(schema='squat-observation-checklist-v2',reviewed_repetitions=len(rows),train_repetitions=sum(r['split']=='train' for r in rows),validation_repetitions=sum(r['split']=='validation' for r in rows),train_people=sorted(train_people),validation_people=sorted(val_people),test_opened=False,expert_validated=False,source_preserved=before==digest(read_source()),model_sha256=hashlib.sha256((O/'candidate.joblib').read_bytes()).hexdigest(),items=reports,limitations=['旧開発資料を再利用した検証で独立最終試験ではない','정면·측면 시각 표본 초안이며 제공자 오류명을 직접 정답으로 쓰지 않음','새 항목은 합산 배점·총점 없음','검증 한 사람·항목별 최대5사례로 일반화 성능 확정 불가'])
report['limitations'][0]='기존 개발 자료 재사용 검증이며 독립 최종 시험이 아님'
(O/'feature-rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));(O/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
