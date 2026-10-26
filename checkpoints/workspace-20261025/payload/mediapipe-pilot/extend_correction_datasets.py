"""Restore all three dataset x three extractor choices, with explicit domains."""
from pathlib import Path
import json,re,csv
import numpy as np
import joblib
from train_squat_correction import C,ROOT,KOREAN,ADVICE,feature_sequence

base=json.loads((C/'demo-predictions.json').read_text())
binary=joblib.load(C/'normal_error.joblib');types=joblib.load(C/'error_type.joblib')
datasets={'REHAB26':{'title':'REHAB26 · 스쿼트','unit':'반복','scope':'학습에서 제외한 Honza의 반복 10개입니다. REHAB26 사람 분리 시험 정확도와 이 시연의 정확도는 다른 수치입니다.','predictions':base['predictions'],'metrics':base['demo_metrics'],'resources':base['resources'],'paths':{m:f'outputs/correction/gpu/{m}/REHAB26/' for m in base['models']},'labels':'반복별 정상·오류·오류 유형 정답 있음','training_domain':True}}
for name,title in [('QEVD','QEVD · 일반 스쿼트'),('Pose-Estimation-Model-Comparison','유선혁님 자료 · 스쿼트')]:
 d={'title':title,'unit':'클립','scope':'REHAB26으로 학습한 교정 분류기를 다른 출처에 시험 적용했습니다. 이 출처로 별도 교정 학습하거나 일반화 성능을 검증한 결과가 아닙니다. 반복 경계 없이 클립 전체를 하나의 입력 구간으로 사용합니다.','predictions':{},'metrics':{},'resources':{},'paths':{},'training_domain':False}
 for m in ['lite','full','heavy']:
  folder=ROOT/'outputs/selected-exercise/gpu'/m/name;points=[json.loads(x) for x in (folder/'landmarks.jsonl').open()];sample=json.loads((folder/'sample-and-labels.json').read_text());mr=json.loads((folder/'metrics.json').read_text());world=[];vis=[]
  for r in points:
   if r['world_landmarks']:world.append(np.array(r['world_landmarks'])[:,:3]);vis.append(np.array(r['normalized_landmarks'])[:,3])
   else:world.append(np.full((33,3),np.nan));vis.append(np.zeros(33))
  feat,coverage=feature_sequence(np.asarray(world),np.asarray(vis));n=len(points);fps=sample['video_metadata']['fps']
  source_labels=sample['labels'] if name=='QEVD' else []
  truth_text='원본 클립 라벨: '+', '.join(source_labels) if source_labels else '정상·오류의 검증된 정답 라벨 없음'
  item={'repetition':1,'first_frame':0,'last_frame':n-1,'start_seconds':0,'end_seconds':n/fps,'ground_truth':None,'ground_truth_type':None,'ground_truth_ko':truth_text,'coverage':coverage,'prediction':'UNKNOWN','prediction_ko':'관측 부족 · 판정 보류','predicted_error_type':None,'error_score':None,'type_score':None,'advice':'관절 가림·누락으로 유효 특징이 부족해 판정을 보류합니다.','label_comparison':'교정 정확도 비교 불가 · 라벨 체계/정답 차이'}
  if feat is not None:
   x=feat.reshape(1,-1);prob=binary.predict_proba(x)[0];label=str(binary.predict(x)[0]);tp=types.predict_proba(x)[0];ty=str(types.predict(x)[0]);score=float(max(tp))
   item.update(prediction=label,prediction_ko='정상 예측' if label=='NORMAL' else '오류 예측',error_score=float(prob[list(binary.classes_).index('ERROR')]),type_score=score)
   if label=='NORMAL':item['advice']=ADVICE['NORMAL']
   elif ty!='NORMAL' and score>=.35:item.update(predicted_error_type=ty,prediction_ko='오류 예측 · '+KOREAN[ty],advice=ADVICE[ty])
   else:item.update(prediction_ko='오류 예측 · 유형 미확정',advice='REHAB26 학습 모델이 오류 패턴을 예측했지만 이 출처의 교정 성능은 검증되지 않았습니다.')
  d['predictions'][m]=[item];d['metrics'][m]={'segments':1,'assessed':int(feat is not None),'normal_error_accuracy':None,'macro_f1':None};d['resources'][m]={'pose_mean_ms':mr['pose_mean_ms'],'frames':n,'source_video':sample['video']};d['paths'][m]=str(folder.relative_to(ROOT))+'/'
 d['labels']=truth_text;datasets[name]=d
payload={**base,'datasets':datasets}
(C/'all-dataset-predictions.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
with (C/'all-dataset-predictions.csv').open('w') as f:
 fields=['dataset','pose_model','repetition','prediction','predicted_error_type','error_score','coverage','ground_truth','ground_truth_type','ground_truth_ko'];w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader()
 for name,d in datasets.items():
  for model,rows in d['predictions'].items():
   for r in rows:w.writerow({'dataset':name,'pose_model':model,**r})

p=ROOT/'selected-exercise.html';page=p.read_text().replace('사람을 분리한 최종 시험 정상·오류 정확도','REHAB26에서 사람을 분리한 시험 정상·오류 정확도')
if 'id="dataset"' not in page:
 page=page.replace('<section><label>관절 추출기','<section><label>데이터셋 <select id="dataset"><option value="REHAB26">REHAB26</option><option value="QEVD">QEVD</option><option value="Pose-Estimation-Model-Comparison">유선혁님 자료</option></select></label> <label>관절 추출기',1)
 page=page.replace('REHAB26 · 실제 지도학습 분류기 · 학습에 쓰지 않은 Honza 영상','데이터셋 3개 × Lite / Full / Heavy · 실제 학습 분류기와 외부 적용 비교')
 page=page.replace('<div class="layout">','<p id="dataset-scope" class="warning"></p><div class="layout">',1)
page=page.replace('<h2>정상·오류 반복별 비교</h2>','<h2 id="comparison-title">정상·오류 반복별 비교</h2>').replace('<th>반복</th>','<th>구간</th>')
page=page.replace('각 행의 재생 버튼은 해당 반복으로 이동합니다. 정답은 학습 입력으로 전달하지 않았습니다.','각 행의 재생 버튼은 해당 구간으로 이동합니다. REHAB26은 반복 단위, 다른 두 자료는 클립 전체 단위입니다. 정답을 모델 입력으로 전달하지 않습니다.')
page=page.replace('반복 경계는 제공자 주석을 사용합니다. 실시간 반복 감지·의료적 교정이 아닙니다.','REHAB26 반복 경계는 주석을 사용하고, 다른 두 자료는 클립 전체를 사용합니다. 실시간 반복 감지·의료적 교정이 아닙니다.')
page=page.replace('href="correction/demo-predictions.csv"','href="correction/all-dataset-predictions.csv"')
js='''const DATA=__DATA__;const video=document.querySelector('#video'),model=document.querySelector('#model'),dataset=document.querySelector('#dataset');const fmt=x=>(x*100).toFixed(1)+'%';document.querySelector('#test').textContent=fmt(DATA.training.results.normal_error.test.accuracy);let selected=[],current=null;function color(r){return r.prediction==='NORMAL'?'normal':r.prediction==='ERROR'?'error':'unknown'}function comparison(r){if(r.ground_truth===null)return r.label_comparison||'정답 없음';if(r.prediction!==r.ground_truth)return '정상·오류 불일치';if(r.ground_truth==='NORMAL')return '정상 일치';return r.predicted_error_type===r.ground_truth_type?'정상·오류 일치 · 유형 일치':'정상·오류 일치 · 유형 불일치/미확정'}function update(){current=DATA.datasets[dataset.value];selected=current.predictions[model.value];const base=current.paths[model.value];video.src=base+'preview.mp4';document.querySelector('#csv').href=base+'angles.csv';document.querySelector('#dataset-scope').textContent=current.title+' — '+current.scope;document.querySelector('#comparison-title').textContent=current.unit==='반복'?'정상·오류 반복별 비교':'클립 전체 시험 적용';const result=current.metrics[model.value],resource=current.resources[model.value];document.querySelector('#stats').textContent=(result.normal_error_accuracy===null?'이 출처의 교정 정확도: 산정하지 않음':'시연 '+result.segments+'반복 정확도 '+fmt(result.normal_error_accuracy))+' · 관절 추론 평균 '+resource.pose_mean_ms.toFixed(2)+'ms · '+resource.frames+'프레임';const body=document.querySelector('#rows');body.replaceChildren();for(const r of selected){const tr=document.createElement('tr');tr.dataset.rep=r.repetition;for(const text of [current.unit+' '+r.repetition,r.ground_truth_ko,r.prediction_ko,comparison(r)]){const td=document.createElement('td');td.textContent=text;tr.appendChild(td)}const td=document.createElement('td'),button=document.createElement('button');button.textContent='재생';button.onclick=()=>{video.currentTime=Math.max(0,r.start_seconds-.2);video.play().catch(()=>{})};td.appendChild(button);tr.appendChild(td);body.appendChild(tr)}tick()}function tick(){if(!current)return;const t=video.currentTime;const done=video.ended;const active=selected.find(r=>r.start_seconds<=t&&t<r.end_seconds);const finished=selected.filter(r=>r.end_seconds<=t||(done&&r===selected.at(-1))).at(-1);const row=active||finished;document.querySelector('#position').textContent=active?current.unit+' '+active.repetition+' 처리 중':finished?current.unit+' '+finished.repetition+' 완료':'첫 구간 전';const prediction=document.querySelector('#prediction');if(finished){prediction.textContent='완료 '+current.unit+' '+finished.repetition+': '+finished.prediction_ko;prediction.className='prediction '+color(finished);document.querySelector('#score').textContent=finished.error_score===null?'관측 부족':'오류 점수 '+finished.error_score.toFixed(3)+' (보정되지 않은 분류기 출력)';document.querySelector('#advice').textContent=finished.advice}else{prediction.textContent=current.unit+' 완료 후 판정';prediction.className='prediction unknown';document.querySelector('#score').textContent='';document.querySelector('#advice').textContent='현재 구간을 끝까지 본 후 분류합니다.'}document.querySelector('#truth').textContent=finished?'원본 주석 — '+finished.ground_truth_ko:'구간 완료 후 원본 주석과 비교합니다';for(const tr of document.querySelectorAll('#rows tr'))tr.classList.toggle('active',!!row&&Number(tr.dataset.rep)===row.repetition)}model.onchange=update;dataset.onchange=update;video.addEventListener('timeupdate',tick);video.addEventListener('ended',tick);update();'''
js=js.replace('__DATA__',json.dumps(payload,ensure_ascii=False).replace('</','<\\/'))
page=re.sub(r'<script>.*?</script>',lambda _: '<script>'+js+'</script>',page,flags=re.S);p.write_text(page)
report=ROOT/'reports/2026-10-16-squat-correction-training.md';text=report.read_text()
text=text.replace('QEVD·Pose Comparison에는 이번 REHAB26 분류기의 일반화가 검증되지 않아 이 화면에서 자동 교정 모델을 적용하지 않는다. 기존 관절 전용 화면을 별도 링크로 유지한다.','QEVD·Pose Comparison은 동일 REHAB26 학습 분류기의 외부 적용을 시험 표시한다. 각 출처별 별도 학습이나 교정 성능 검증을 완료한 것은 아니다. 이전 관절 전용 화면도 별도 링크로 유지한다.')
heading='## 9. 데이터셋 3개 × 추출기 3개 선택'
if heading not in text:
 text+='\n'+heading+'\n\nREHAB26·QEVD·Pose Comparison을 선택하고 각각 Lite/Full/Heavy를 선택할 수 있게 했다. REHAB26은 학습에서 제외한 사람의 10개 반복을 비교한다. 다른 두 자료는 기존 추출 결과에 같은 교정 분류기를 적용한 클립 단위 시험이며, 출처별 교정 모델을 새로 학습한 것이 아니다. QEVD 라벨은 원문 그대로 표시하고 REHAB26 오류 유형과 동일 정답이라고 간주하지 않는다. Pose Comparison은 정상·오류 정답이 없어 정확도를 산정하지 않는다. 가림 등으로 특징 관측률이 부족하면 판정을 보류한다.\n'
report.write_text(text)
print(json.dumps({name:{m:d['predictions'][m][0]['prediction_ko'] for m in ['lite','full','heavy']} for name,d in datasets.items()},ensure_ascii=False,indent=2))

# Preserve measured resource cards and charts when regenerating dataset choices.
if (ROOT/'correction/resource-dashboard.json').exists():
 import runpy
 runpy.run_path(str(ROOT/'add_resource_dashboard.py'),run_name='__main__')
