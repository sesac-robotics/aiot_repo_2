"""검증 결과와 직접 검토 기록을 보존하고 집계 메타데이터만 팀 번들에 저장한다."""
import datetime
import hashlib
import html
import json
from pathlib import Path

BASE=Path('/home/jayden/SeSACPAI-Mid')
ROOT=BASE/'output/five-exercise-recovery-20261023'
BUNDLE=BASE/'team-posture-scoring/Server/AIService/artifacts/posture_score'
def read(name):return json.loads((ROOT/name).read_text())
def save(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
comparison=read('comparison.json')
qualification=read('label_qualification.json')
reserve=read('final_test_hashed_manifest.json')
trials={name:read(name+'.json') for name in ('plank_shoulder_trial','plank_compact_trial')}
summary=dict(schema='posture-score-development-round-v1',date='2026-10-23',label_origin='direct_visual_draft',expert_validated=False,external_discovery=False,qualification=qualification,final_test_reservation=dict(candidate_count=len(reserve['records']),counts={**reserve['counts'],'shoulder':0},source_hashes_verified=True,person_identity_verified=False,pixels_or_score_targets_opened=False,final_test_executed=False),selection=dict(fitting='train_only_group_balanced_cv',promotion_uses_reused_development_validation=True,minimum_total_mae_improvement_points=1,error_item_mae_must_not_increase=True,independent_performance_estimate=False),exercises=[{k:r[k] for k in ('exercise','baseline','raw_refit','accepted','ai_adopted','rule_adopted')} for r in comparison['exercises']],rejected_plank_geometry_trials={k:{field:r[field] for field in ('baseline','candidate','adopted')} for k,r in trials.items()},release_weights_changed=False,release_rules_changed=False,validation=dict(pytest_passed=40,pytest_subtests_passed=13,mypy_source_files=25,black_checked_files=25,pylint_errors_only=True,numeric_parity_cases=91,numeric_parity_matches=91,parity_is_training_replay_not_accuracy_test=True))
save(BUNDLE/'development_round3.json',summary)
save(ROOT/'summary.json',summary)
receipt=dict(schema='direct-label-freeze-v1',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),direct_draft_rows=25,expert_validated=False,not_rule_generated_targets=True,files={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('manual_decisions.json','labels.json','subject_tracking_review.json','new_feature_rows.json','final_test_hashed_manifest.json','post_label_partition_audit.json','comparison.json')},training_and_validation_only=True,final_pixels_or_score_targets_opened=False,release_weights_unchanged=True)
save(ROOT/'label_freeze_receipt_final.json',receipt)
names={'pushups':'푸시업','squats':'스쿼트','shoulder':'숄더 프레스','plank':'플랭크','ytw':'Y–T–W'}
rows=''
for r in summary['exercises']:
 b=r['baseline'];c=r['raw_refit'];a=r['accepted'];e=r['exercise']
 rows+=f"<tr><td>{names[e]}</td><td>{qualification['effective_new_training_rows'].get(e,0)}</td><td>{b['total_cases']}</td><td>{b['AI_MAE']:.2f}</td><td>{c['AI_MAE']:.2f}</td><td>{a['AI_MAE']:.2f}</td><td>{a['Rule_MAE']:.2f}</td><td>기존 유지</td></tr>"
links=('labels.json','manual_decisions.json','label_qualification.json','subject_tracking_review.json','comparison.json','plank_shoulder_trial.json','plank_compact_trial.json','label_freeze_receipt_final.json','release_parity.json','release_demo/index.html')
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>5종 자세 점수 보강 · 2026-10-23</title><style>body{font:16px/1.65 system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#182638;background:#f5f7fa}table{border-collapse:collapse;background:white;width:100%;font-variant-numeric:tabular-nums}td,th{border:1px solid #ccd3dc;padding:10px;text-align:left}a{color:#174eb6}.note{background:#fff3ce;padding:16px;border-radius:8px}</style><h1>5종 자세 점수 보강 결과</h1><p>보유 자료 43개 원본 선별 → 25개 구간 직접 시각 라벨 초안 → 관측 조건을 충족한 21개 학습 입력. 라벨은 AI가 장면을 검토한 실험 초안이며 전문가 정답이 아닙니다. MediaPipe Heavy는 관절 추출기이고 점수 모델은 별도입니다.</p><p class="note">5종 재학습 후보와 YTW 규칙 보정을 비교했으나 채택 조건을 통과하지 못했습니다. 배포 모델·규칙은 기존 버전을 유지합니다. 아래 오차는 반복 사용한 개발 검증 자료의 결과이며 독립 최종 시험 정확도가 아닙니다. 관측 불가 항목은 점수로 채우지 않습니다.</p><table><thead><tr><th>운동</th><th>신규 유효 입력</th><th>총점 비교 구간</th><th>기존 AI MAE</th><th>재학습 후보 AI MAE</th><th>유지 AI MAE</th><th>유지 Rule MAE</th><th>결정</th></tr></thead><tbody>'''+rows+'''</tbody></table><p>총점 MAE 단위는 100점 중 점수입니다. 표의 구간 수는 참여자 수가 아닙니다. 서로 다른 관측 항목을 임의 합산하지 않았습니다. 학습은 참여자·원본·촬영 연결 그룹을 분리한 학습 자료 내부 CV로 수행했고, 배포 후보 선택에는 기존 개발 검증을 사용했습니다.</p><h2>오류 분석과 남은 한계</h2><p>플랭크의 어깨 높이 특징 추가 후보는 MAE가 10.26→10.48로 악화되어 제외했습니다. 3개 특징으로 단순화한 후보는 총점 비교 2개 정상 구간에서 2.82까지 낮아졌지만, 오류 항목의 등급 MAE가 0.448→0.741로 악화되어 제외했습니다. 오류 판별을 희생해 정상 점수만 높이는 모델은 채택하지 않았습니다.</p><p>플랭크 신규 입력 2개는 바닥 지지 대비 사례 1개와 약 0.5초의 짧은 유지 구간 1개입니다. 전자는 운동 실패 총점 정답이 아니며 후자는 3초 유지 완료가 아닙니다. YTW는 Y/T 단계 혼합과 투영 구도 제약이 남아 있습니다. 숄더 프레스의 일부 가림은 해당 항목만 미판정 처리했습니다.</p><h2>최종 시험 분리</h2><p>후보 991개(푸시업 263·스쿼트 325·플랭크 260·YTW 143)는 원본 지문과 참여자 메타데이터로 예약했습니다. 영상·점수 라벨은 열지 않았습니다. 실제 참여자 중복 확인이 남아 있으며 숄더 프레스의 미노출 시험 자료는 확보되지 않았습니다. 이 수를 학습 가능한 라벨 수나 완료한 최종 시험 수로 표현하면 안 됩니다.</p><h2>검토 자료</h2><ul>'''+''.join(f'<li><a href="{html.escape(n)}">{html.escape(n)}</a></li>' for n in links)+'''</ul><p>분할 방어·재학습 도구와 항목별 학습 등급 분포 표시를 팀 소스에 반영했습니다. 40개 테스트·13개 하위 검사, mypy 25개 파일, Black 25개 파일, Pylint 오류 검사를 통과했습니다. 기존 학습 관절 91개 재실행 점수는 모두 일치하며 이는 정확도 검증이 아닙니다.</p></html>'''
(ROOT/'index.html').write_text(page,encoding='utf-8')
print(json.dumps({'report':str(ROOT/'index.html'),'metadata':str(BUNDLE/'development_round3.json')}))
