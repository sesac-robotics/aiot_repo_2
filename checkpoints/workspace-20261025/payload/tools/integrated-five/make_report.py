import collections,datetime,platform,time
import numpy as np,sklearn,joblib
from common import B,O,read,save,sha,status

def main():
 started=time.monotonic()
 while '"stage": "native_complete"' not in (O/'native-balanced-training.log').read_text():
  if time.monotonic()-started>7200:raise TimeoutError('Balanced native training not finished')
  time.sleep(5)
 while 'AIHUB_OBSERVABLE_COMPLETE' not in (O/'native-aihub-observable.log').read_text():
  if time.monotonic()-started>7200:raise TimeoutError('Observable AIHub training unfinished')
  time.sleep(5)
 while 'FINAL_SCORE_REFINEMENT_COMPLETE' not in (O/'score-refinement.log').read_text():
  if time.monotonic()-started>7200:raise TimeoutError('Final score selection unfinished')
  time.sleep(5)
 native_inputs=read(O/'qevd-native-inputs.json')
 for r in native_inputs:r['cache_sha256']=sha(r['cache'])
 save('qevd-native-inputs.json',native_inputs)
 rehab=read(O/'rehab26-native-training.json');rehab.pop('historical_test_not_reopened',None);rehab['historical_test_evaluated']=False;save('rehab26-native-training.json',rehab)
 score=read(O/'score-training.json');native={s:read(O/f'{s}-native-training.json') for s in ['qevd','rehab26','aihub']};rows=read(O/'score-inputs.frozen.json');audit=read(O/'score-intake-audit.json');neural=read(O/'return-neural-comparison.json');sources=[]
 for p in [B/'data/r2d2-team-20261023/annotations/labels.frozen.json',B/'output/squat-return-20261023/labels.frozen.json',B/'output/five-exercise-recovery-20261022/held_round2/candidate_input_manifest.json',B/'output/five-exercise-recovery-20261022/squat_partial_top_trial/candidate_input_manifest.json',B/'output/five-exercise-recovery-20261023/new_feature_rows.json',O/'score-inputs.frozen.json',O/'qevd-native-inputs.json',O/'aihub-native-inputs.json']:
  sources.append(dict(path=str(p),sha256=sha(p)))
 models=[dict(path=str(p.relative_to(B)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(O.rglob('*.joblib')) if 'rejected-auxiliary' not in p.parts]
 fitted=joblib.load(O/'score-candidates.joblib');actual_ids={identifier for h in fitted['heads'].values() for identifier in h['trained_ids']};validation_ids={p['id'] for h in score['heads'] for p in h.get('validation_predictions',[])}
 support=[]
 for ex in ['squats','pushups','shoulder','plank','ytw']:
  rs=[r for r in rows if r['exercise']==ex];hs=[r for r in score['heads'] if r['key'][0]==ex];support.append(dict(exercise=ex,segments=len(rs),people=len({r['person'] for r in rs}),training_segments=sum(r['id'] in actual_ids for r in rs),validation_segments=sum(r['id'] in validation_ids for r in rs),trained_heads=sum(r['status']=='trained_experimental' for r in hs),gate_passed=sum(r.get('candidate_gate_passed',False) for r in hs),unsupported_heads=[r['key'][1:] for r in hs if r['status']!='trained_experimental']))
 summary=dict(date='2026-10-24',status='completed_experimental_training',exercises=support,score_input_count=len(rows),actual_score_training_segments=len(actual_ids),source_families_actually_fit=dict(collections.Counter(r['family'] for r in rows if r['id'] in actual_ids)),source_families=audit['source_families'],native_tasks={s:len(d['tasks']) for s,d in native.items()},models=models,input_provenance=sources,versions=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__),training_method='160-tree regressors/classifiers and 80-iteration boosting candidates. Neural height-return comparison uses up to 80 epochs and person-separated stopping.',neural_epochs=neural['full_training_epochs'],production_models_replaced=False,final_test_used=False,limitations=['Project score labels remain nonexpert visual drafts.','All comparisons are development evaluation; no new independent final accuracy claimed.','AI-Hub training uses supplied 2D joints and conditions; original training-image matching and pixel-network training are not complete.','No confirmed severe standing-return grade 0 examples.','Not every criterion has sufficient normal/error participant groups.','Runtime remains on existing models; candidate gates do not establish prospective performance.'])
 save('summary.json',summary)
 import json
 dashboard=dict(summary=summary,score=score,native=native,neural=neural,intake=audit);save('report-data.json',dashboard)
 # Human-readable results persist after shutdown.
 lines=['통합 5종 운동 학습 결과 — 2026-10-24','',f'점수 입력: {len(rows)}구간. 출처별 수는 아래와 같으며 독립 사람 수가 아니다.',json.dumps(audit['source_families'],ensure_ascii=False),'',f'실제 점수 학습 항목: {score["trained_heads"]}, 내부 추가 검증 조건 통과: {score["gate_passed"]}. 기존 서비스 모델은 교체하지 않았다.','QEVD·REHAB26·AI-Hub 원본 라벨 과제는 프로젝트 점수와 구분해서 학습했다.','',f'복귀 신경망 전체 재학습: {neural["full_training_epochs"]}에포크. 교차검증에서는 각 폴드 내부 사람 분리로 조기 종료했다.','', '운동별 입력·학습 현황:',json.dumps(support,ensure_ascii=False,indent=2),'','남은 한계:']+summary['limitations']+['','재개: 같은 프로젝트에서 tools/integrated-five/의 준비·학습 스크립트와 summary.json을 읽는다. 캐시와 모델을 보존했다.','화면: http://127.0.0.1:8801/output/integrated-five-20261024/index.html','로컬 서버 재시작: python3 -m http.server 8801 --bind 127.0.0.1 --directory /home/jayden/SeSACPAI-Mid']
 (O/'RESULTS.txt').write_text('\n'.join(lines))
 status('results_saved',models=len(models))
if __name__=='__main__':main()
