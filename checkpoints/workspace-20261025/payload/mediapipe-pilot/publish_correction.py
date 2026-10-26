from pathlib import Path
import json,csv,time,shutil,hashlib
import numpy as np
import joblib
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from train_squat_correction import C,ROOT,CLASSES,KOREAN,ADVICE,feature_sequence

training=json.loads((C/'training-results.json').read_text());demo=json.loads((C/'demo-samples.json').read_text())[0]
binary=joblib.load(C/'normal_error.joblib');types=joblib.load(C/'error_type.joblib');predictions={};metrics={};view_resources={}
annotations=[x for x in demo['labels'] if int(x['last_frame'])<1530]
assert len(annotations)==10
for model in ['lite','full','heavy']:
 folder=ROOT/'outputs/correction/gpu'/model/'REHAB26';world=[];visibility=[]
 for line in (folder/'landmarks.jsonl').open():
  row=json.loads(line)
  if row['world_landmarks']:
   world.append(np.array(row['world_landmarks'])[:,:3]);visibility.append(np.array(row['normalized_landmarks'])[:,3])
  else:world.append(np.full((33,3),np.nan));visibility.append(np.zeros(33))
 world=np.asarray(world);visibility=np.asarray(visibility);rows=[]
 for a in annotations:
  first,last=int(a['first_frame']),int(a['last_frame']);feat,coverage=feature_sequence(world[first:last+1],visibility[first:last+1]);truth='NORMAL' if a['correctness_score']=='1' else 'ERROR';true_type='NORMAL' if truth=='NORMAL' else a['error_description']
  item={'repetition':int(a['repetition_number']),'first_frame':first,'last_frame':last,'start_seconds':first/30,'end_seconds':(last+1)/30,'ground_truth':truth,'ground_truth_type':true_type,'ground_truth_ko':KOREAN[true_type],'coverage':coverage,'prediction':'UNKNOWN','prediction_ko':'관측 부족','predicted_error_type':None,'error_score':None,'type_score':None,'advice':'관절 관측이 부족해 판정을 보류합니다.'}
  if feat is not None:
   X=feat.reshape(1,-1);prob=binary.predict_proba(X)[0];error_score=float(prob[list(binary.classes_).index('ERROR')]);label=str(binary.predict(X)[0]);tp=types.predict_proba(X)[0];type_label=str(types.predict(X)[0]);type_score=float(max(tp))
   item.update(prediction=label,prediction_ko='정상 예측' if label=='NORMAL' else '오류 예측',error_score=error_score,type_score=type_score)
   if label=='NORMAL':item['advice']=ADVICE['NORMAL']
   elif type_label!='NORMAL' and type_score>=.35:item.update(predicted_error_type=type_label,prediction_ko='오류 예측 · '+KOREAN[type_label],advice=ADVICE[type_label])
   else:item.update(prediction_ko='오류 예측 · 유형 미확정',advice='오류 패턴으로 분류했지만 구체적인 교정 유형은 확정하지 못했습니다.')
  rows.append(item)
 valid=[r for r in rows if r['prediction']!='UNKNOWN']
 metrics[model]={'segments':len(rows),'assessed':len(valid),'normal_error_accuracy':accuracy_score([x['ground_truth'] for x in valid],[x['prediction'] for x in valid]) if valid else None,'macro_f1':f1_score([x['ground_truth'] for x in valid],[x['prediction'] for x in valid],average='macro',zero_division=0) if valid else None,'confusion_matrix_normal_error':confusion_matrix([x['ground_truth'] for x in valid],[x['prediction'] for x in valid],labels=['NORMAL','ERROR']).tolist() if valid else []}
 predictions[model]=rows
 rs=json.loads((ROOT/'outputs/correction/gpu'/model/'resources.json').read_text());mr=json.loads((folder/'metrics.json').read_text())
 view_resources[model]={'pose_mean_ms':mr['pose_mean_ms'],'peak_process_rss_mib':max(x['process_rss_mb'] for x in rs),'source_video':demo['video'],'frames':mr['frames']}
 # Demonstration scores are not used to select/retrain the classifiers.
payload={'training':training,'predictions':predictions,'demo_metrics':metrics,'resources':view_resources,'video_id':demo['video_id'],'models':['lite','full','heavy'],'label_names':KOREAN}
(C/'demo-predictions.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
with (C/'demo-predictions.csv').open('w') as f:
 fields=['pose_model','repetition','first_frame','last_frame','ground_truth','ground_truth_type','prediction','predicted_error_type','error_score','coverage'];w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader()
 for m,rows in predictions.items():
  for r in rows:w.writerow({'pose_model':m,**r})

lines=['# 스쿼트 정상·오류 교정 분류기 학습 보고서','','작성일: 2026-10-16 · 실제 지도학습 완료 · 실험용, 교정 성능 미충족','','## 1. 결과 요약','','이번에는 관절 추출에 더해 정상·오류 라벨로 분류기를 실제 학습했다. 다만 보지 않은 사람의 정상·오류 정확도가 약 54%로 낮다. 정확한 교정 기능이 완성됐다고 주장하지 않으며, 화면에 모델 예측과 실제 주석을 분리해 보여준다. 오류별 안내 문구는 학습된 오류 유형 예측에 연결한 고정 문구이며 생성형 코칭이나 학습된 문장 생성기가 아니다.','',
'## 2. 학습 자료와 분할','','- REHAB26 스쿼트의 554개 고유 반복(정상 276, 오류 278), 카메라 3개로 총 1,662개 반복-카메라 표본.',
'- 학습: Petra·Terka·Vasil 999표본. 모델 선택: Mima 333표본. 최종 평가: Honza 330표본. 모든 카메라·반복을 사람 단위로 묶어 분리했다.',
'- correctness_score=1은 정상, 5는 오류. 오류는 무릎 안쪽 모임·몸통 옆 기울기·깊이 부족·무릎 전방 이동·뒤꿈치 들림 5종이다. Segmentation.txt의 0-based inclusive 경계를 사용했다.',
'- 제공된 MediaPipe 3D 관절 좌표로 학습했다. 이 파일의 추출기 버전/크기가 이번 Lite/Full/Heavy와 동일하다고 확인된 것은 아니다. 새 추출기에 대한 10반복 시연 점수를 별도로 측정했다.',
'- 반복 완료 구간 전체의 골반 중심 정규화 좌표·관절 각도·거리 비율에서 분위수와 상대 시간 16지점 특징을 계산했다. 사람 이름·카메라 ID·정답·절대 프레임 번호·반복 번호는 모델 입력에서 제외했다.',
'- visibility<0.5 관절은 결측으로 처리했다. 반복 내부 특징 계산에서 시간 보간, 모델 파이프라인에서 학습 자료로만 적합한 중앙값 대체를 사용했다. 원본 좌표를 덮어쓰지 않았다. 관측 유효율 60% 미만은 판정 보류다.',
'- ExtraTrees 300개 트리의 정상/오류 이진 분류기와 6클래스 오류 유형 분류기 2개를 학습했다. min_samples_leaf 1/3/7 중 validation macro-F1으로 선택했다. test는 선택 후 1회 평가했고 시연 결과로 모델을 다시 선택하지 않았다.',
'', '## 3. 실제 학습 시간','','전체 자료 준비·특징 생성·후보 학습·평가에 '+f"{training['total_preparation_and_training_seconds']:.1f}초"+'가 걸렸다. 제공된 좌표를 재사용하고 작은 트리 모델을 CPU 4스레드로 학습했기 때문에 짧다. 영상 전체의 새 관절 추출이나 딥러닝 전체 학습 시간이 아니다. 이전 2~4시간은 데이터 연결·시연·검토를 포함한 작업 예산이었다.','',
'## 4. 사람 분리 평가','','| 모델 | validation macro-F1 | test 정확도 | test balanced accuracy | test macro-F1 |','|---|---:|---:|---:|---:|']
for task,k in [('normal_error','정상·오류'),('error_type','정상+오류 5종')]:
 r=training['results'][task];lines.append(f"| {k} | {r['validation']['macro_f1']:.3f} | {r['test']['accuracy']*100:.1f}% | {r['test']['balanced_accuracy']:.3f} | {r['test']['macro_f1']:.3f} |")
lines+=['','validation과 test 차이가 커 사람 간 일반화가 부족하다. 정상·오류 test 표본은 165개씩으로 균형적이며, 현재 약 54% 정확도는 실사용 근거로 충분하지 않다. 3개 카메라 표본은 동일 반복의 다른 시점이므로 330개 독립 사람 실험으로 해석하지 않는다.','',
'## 5. 화면 시연: 완전히 보류한 Honza','','Honza cam1의 첫 1,530프레임(51초)을 새 GPU Lite/Full/Heavy로 추출하고 학습된 동일 분류기에 입력했다. 세 가지 별도 교정 분류기를 학습한 것이 아니라 관절 추출기만 바꾼 것이다. 원본 주석에서 정상 5개와 오류 5개, 총 10개의 완료 반복을 사용했다.','',
'| 새 관절 추출기 | 판정 가능 반복 | 정상·오류 정확도 | macro-F1 | 평균 관절 추론 ms |','|---|---:|---:|---:|---:|']
for m,r in metrics.items():lines.append(f"| {m} | {r['assessed']}/{r['segments']} | {100*r['normal_error_accuracy']:.1f}% | {r['macro_f1']:.3f} | {view_resources[m]['pose_mean_ms']:.2f} |")
lines+=['','시연 10개 반복은 위 test 사람의 일부이므로 별도 독립 검증셋이 아니다. 정답 주석으로 반복 경계를 공급했으며 자동 반복 검출을 구현한 것은 아니다. 반복 전체 특징을 쓰므로 결과는 해당 반복이 끝난 시점부터 표시한다. 프레임별 실시간 정상/오류 모델이라고 표시하지 않는다.','',
'## 6. 화면에서 구분하는 정보','','- 모델 판정: 실제 학습 분류기의 정상/오류 예측.', '- 교정 안내: 모델이 오류로 예측하고 오류 유형 점수가 기준 이상일 때만 해당 고정 안내 표시. 종류를 확정하지 못하면 유형 미확정 표시.', '- 실제 주석: 제공자가 붙인 반복의 정상/오류·오류 유형. 모델 출력으로 위장하지 않는다.', '- 오류 점수: 보정되지 않은 분류기 출력이며 정확한 확률·의료적 신뢰도가 아니다.', '- QEVD·Pose Comparison에는 이번 REHAB26 분류기의 일반화가 검증되지 않아 이 화면에서 자동 교정 모델을 적용하지 않는다. 기존 관절 전용 화면을 별도 링크로 유지한다.','',
'## 7. 재현과 산출물','','- 학습 코드: ../train_squat_correction.py', '- 저장한 정상·오류 모델: ../correction/normal_error.joblib', '- 저장한 오류 유형 모델: ../correction/error_type.joblib', '- 사람별 분할·표본: ../correction/sample-manifest.json', '- 전체 지표·혼동행렬: ../correction/training-results.json', '- 새 추출기의 실제 예측: ../correction/demo-predictions.csv', '- 학습에 사용한 특징: ../correction/training-features.npz', '- 시연 화면: ../selected-exercise.html',
'', '모델 파일은 이 실험에서 직접 생성한 로컬 파일만 로드한다. T9 원본과 정식 팀 저장소는 변경하지 않았다. 재실행은 가상환경에서 train_squat_correction.py → pilot.py(각 모델, correction/demo-samples.json, run-name correction, max-frames 1530) → publish_correction.py 순서다.','',
'## 8. 다음 단계','','이 모델은 성능이 낮은 학습 경로 검증용 기준 모델이다. 반복 경계·촬영 방향·수행자별 관절 분포·라벨별 특징을 검토하고, 다음 실험은 사람별 교차 검증을 사전에 정해 비교해야 한다. 이미 확인한 Honza 점수에 맞춰 튜닝한 뒤 같은 자료를 미사용 최종 시험이라고 부르면 안 된다. 카메라·실제 사용자 조건의 새 시험 자료 확보 전 정확한 실시간 교정으로 배포하지 않는다.','',
'## 출처','','- [REHAB26 제공 설명](https://zenodo.org/records/23018128)', '- T9/REHAB26/docs/Segmentation.txt, raw/annotations/Segmentation.csv 및 원 제공 MediaPipe NPY.','']
report=ROOT/'reports/2026-10-16-squat-correction-training.md';report.write_text('\n'.join(lines))
if not (ROOT/'pose-only-reference.html').exists():shutil.copy2(ROOT/'selected-exercise.html',ROOT/'pose-only-reference.html')
html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>스쿼트 교정 모델 · 실제 학습 결과</title><style>*{box-sizing:border-box}body{font-family:system-ui;background:#0f1720;color:#e9f0f5;margin:0;padding:30px;max-width:1300px;margin:auto}h1{font-size:28px}.muted{color:#b0c4d2}a{color:#71d9d0}.warning{padding:18px;border:1px solid #d99052;background:#35291f;border-radius:12px;line-height:1.7}section{background:#192a36;border-radius:14px;padding:22px;margin:20px 0}.layout{display:grid;grid-template-columns:minmax(0,2fr) minmax(260px,1fr);gap:20px}video{width:100%;max-height:700px;background:black}select,button{font-size:16px;padding:10px;border-radius:8px;background:#253e50;color:white;border:1px solid #658498;cursor:pointer}.prediction{font-size:25px;line-height:1.5;font-weight:bold}.normal{color:#71dfb0}.error{color:#ffb18f}.unknown{color:#d5dce1}table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:12px;border-bottom:1px solid #38505e;text-align:left}tr.active{background:#284d5e}small{display:block;color:#bbccd7;line-height:1.6}#advice{font-size:18px;line-height:1.7}#truth{padding:12px;background:#253a48;border-radius:8px}@media(max-width:900px){.layout{grid-template-columns:1fr}body{padding:14px}}</style><h1>스쿼트 정상·오류 교정 모델</h1><p class="muted">REHAB26 · 실제 지도학습 분류기 · 학습에 쓰지 않은 Honza 영상</p><div class="warning"><b>실험용 · 아직 교정 성능이 충분하지 않습니다.</b><br>사람을 분리한 최종 시험 정상·오류 정확도 <b id="test"></b>. 모델 예측과 제공자 정답을 구분해서 확인하세요.</div><section><label>관절 추출기 <select id="model"><option value="lite">MediaPipe Lite</option><option value="full">MediaPipe Full</option><option value="heavy">MediaPipe Heavy</option></select></label><p class="muted">교정 분류기는 제공된 MediaPipe 좌표로 학습한 동일 모델입니다. 위 선택은 새 영상의 관절 추출기만 바꿉니다.</p><div class="layout"><video id="video" controls playsinline></video><aside><p id="position"></p><div id="prediction" class="prediction unknown">반복 완료를 기다립니다</div><p id="score" class="muted"></p><p id="advice">재생 후 반복이 끝나면 학습 모델의 판정을 보여줍니다.</p><p id="truth"></p><small>판정은 전체 반복을 본 뒤 표시됩니다. 반복 경계는 제공자 주석을 사용합니다. 실시간 반복 감지·의료적 교정이 아닙니다.</small><p><a id="csv">각도 CSV</a> · <a href="correction/demo-predictions.csv">실제 판정 CSV</a></p></aside></div><p id="stats" class="muted"></p></section><section><h2>정상·오류 반복별 비교</h2><p class="muted">각 행의 재생 버튼은 해당 반복으로 이동합니다. 정답은 학습 입력으로 전달하지 않았습니다.</p><table><thead><tr><th>반복</th><th>제공자 정답</th><th>실제 모델 예측</th><th>정상·오류 / 유형 비교</th><th>이동</th></tr></thead><tbody id="rows"></tbody></table></section><section><h2>학습·평가 범위</h2><p>학습: Petra·Terka·Vasil 999표본 / 모델 선택: Mima 333표본 / 최종 시험: Honza 330표본</p><p>정상·오류 분류기와 오류 유형 분류기를 실제 학습했습니다. 5개 오류는 무릎 안쪽 모임·몸통 옆 기울기·깊이 부족·무릎 전방 이동·뒤꿈치 들림입니다.</p><p>여러 카메라의 동일 반복이 포함되어 있으며 표본 수가 서로 다른 사람 수를 뜻하지 않습니다. 위 10개 반복 시연 점수는 최종 시험 사람의 일부로, 별도 독립 성능 지표가 아닙니다.</p><p><a href="reports/2026-10-16-squat-correction-training.md">학습 보고서</a> · <a href="correction/training-results.json">전체 지표·혼동행렬</a> · <a href="pose-only-reference.html">이전 데이터셋 3개 관절 추출 전용 화면</a></p></section><script>const DATA=__DATA__;const video=document.querySelector('#video'),model=document.querySelector('#model');const fmt=x=>(x*100).toFixed(1)+'%';document.querySelector('#test').textContent=fmt(DATA.training.results.normal_error.test.accuracy);let selected=[];function color(r){return r.prediction==='NORMAL'?'normal':r.prediction==='ERROR'?'error':'unknown'}function update(){selected=DATA.predictions[model.value];video.src='outputs/correction/gpu/'+model.value+'/REHAB26/preview.mp4';document.querySelector('#csv').href='outputs/correction/gpu/'+model.value+'/REHAB26/angles.csv';const score=DATA.demo_metrics[model.value];document.querySelector('#stats').textContent='시연 10반복 정상·오류 정확도 '+fmt(score.normal_error_accuracy)+' · 관절 추론 평균 '+DATA.resources[model.value].pose_mean_ms.toFixed(2)+'ms · 시험 영상 '+DATA.video_id;const body=document.querySelector('#rows');body.replaceChildren();for(const r of selected){const tr=document.createElement('tr');tr.dataset.rep=r.repetition;for(const text of [r.repetition,r.ground_truth_ko,r.prediction_ko,r.prediction!==r.ground_truth?'정상·오류 불일치':r.ground_truth==='NORMAL'?'정상 일치':r.predicted_error_type===r.ground_truth_type?'정상·오류 일치 · 유형 일치':'정상·오류 일치 · 유형 불일치/미확정']){const td=document.createElement('td');td.textContent=text;tr.appendChild(td)}const td=document.createElement('td'),button=document.createElement('button');button.textContent='재생';button.onclick=()=>{video.currentTime=Math.max(0,r.start_seconds-.2);video.play().catch(()=>{})};td.appendChild(button);tr.appendChild(td);body.appendChild(tr)}tick()}function tick(){const t=video.currentTime,active=selected.find(r=>r.start_seconds<=t&&t<r.end_seconds),finished=selected.filter(r=>r.end_seconds<=t).at(-1),r=active||finished;document.querySelector('#position').textContent=active?'반복 '+active.repetition+' 수행 중':finished?'반복 '+finished.repetition+' 완료':'첫 반복 전';const prediction=document.querySelector('#prediction');if(finished){prediction.textContent='완료 반복 '+finished.repetition+': '+finished.prediction_ko;prediction.className='prediction '+color(finished);document.querySelector('#score').textContent=finished.error_score===null?'관측 부족':'오류 점수 '+finished.error_score.toFixed(3)+' (보정되지 않은 분류기 출력)';document.querySelector('#advice').textContent=finished.advice}else{prediction.textContent='반복 완료 후 판정';prediction.className='prediction unknown';document.querySelector('#score').textContent='';document.querySelector('#advice').textContent='현재 반복을 끝까지 본 후 정상·오류를 분류합니다.'}document.querySelector('#truth').textContent=finished?'제공자 정답 — 완료 반복 '+finished.repetition+': '+finished.ground_truth_ko:'반복 완료 후 제공자 정답과 비교합니다';for(const tr of document.querySelectorAll('#rows tr'))tr.classList.toggle('active',!!r&&Number(tr.dataset.rep)===r.repetition)}model.onchange=update;video.addEventListener('timeupdate',tick);update();</script></html>'''
(ROOT/'selected-exercise.html').write_text(html.replace('__DATA__',json.dumps(payload,ensure_ascii=False).replace('</','<\\/')))
print(json.dumps({'training_test':training['results']['normal_error']['test']['accuracy'],'demo':metrics,'report':str(report)},ensure_ascii=False,indent=2))

# Keep the three-dataset selector when regenerating the correction UI.
import runpy
runpy.run_path(str(ROOT/'extend_correction_datasets.py'),run_name='__main__')
