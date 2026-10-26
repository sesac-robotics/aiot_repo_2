"""Labels come from chronological originals, before new pose fitting."""
import json,sys,datetime
from pathlib import Path
B=Path(__file__).resolve().parents[2];O=B/'output/return-label-expansion-v3-20261024'
sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import POLICY,assert_training_allowed
rs=json.loads((O/'intake.frozen.json').read_text())['records'];by={r['review_id']:r for r in rs}
specs=[('E06',.2,1.45,2,[36,39,42]),('E06',1.45,2.65,2,[72,75,78]),('E12',.2,1.3,1,[33,34,35,36]),('E12',1.3,2.1,1,[58,59,60,61,62])]
labels=[]
for n,a,b,g,tops in specs:
 r=by[n];sheets=[O/(n+'-dense.jpg')]+([O/'E12-top-every-frame.jpg'] if n=='E12' else [])
 d=dict(r,id=f"return_v3_{n}_{round(a*1000):06}_{round(b*1000):06}",start=a,end=b,grades=[None,None,g,None],return_grade=g,criterion='return_extension',target_semantics='visible_arm_extension_at_post_descent_upper_turn_not_standing_height',view='front',variant='visible_arm_extension_only',family='QEVD',whole_attempt_observed=True,train_eligible=True,expert_verified=False,annotation_status='chronological_10Hz_and_upper_turn_every_frame_visual_draft_nonexpert',review_top_frames=tops,evidence_sheets=[dict(path=str(p),sha256=sha(p)) for p in sheets],evidence_note_ko=('상승 상단의 매 프레임에서도 팔꿈치 굽힘이 남고 다음 하강이 이어진다. 실제 상승은 있어 심한 미복귀0점과 구분한 부분복귀1점 초안. 초기 지지 자세 자체도 덜 펴져 있어 정상 준비 자세 학습에는 사용하지 않는다.' if g==1 else '하강 뒤 상단에서 팔이 거의 펴진 지지 자세가 관찰되고 다음 하강으로 이어진다. 머리/몸통의 별도 문제를 복귀 오류로 복사하지 않는다.'),correct_initial_state_label=False,source_role_unchanged=True)
 assert sha(d['path'])==d['source_sha256'];labels.append(d)
assert_training_allowed(labels)
decisions=[]
for r in rs:
 n=r['review_id']
 reason='screen_only_no_new_failure_established'
 if r['exercise']=='squats':reason='overview shows initial standing and later recovery or an unfinished last descent; no confirmed new standing-height failure; dense review needed before any new grade'
 if n in ['E08','E10']:reason='near-extension/projection boundary held; no forced grade1'
 if n in ['E06','E12']:reason='four reviewed complete intervals recorded separately; no whole-person or whole-source quality assignment'
 decisions.append(dict(id=r['id'],review_id=n,person=r['person'],split=r['split'],source_sha256=r['source_sha256'],source_grade=None,decision=reason,overview_sha256=sha(O/(n+'-overview.jpg'))))
p=O/'labels.frozen.json';assert not p.exists();p.write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy_sha256=sha(POLICY),reviewed=labels,source_screens=decisions,new_partial_failure_intervals=2,new_normal_return_intervals=2,new_failure_people=1,new_severe_grade0=0,new_squat_failure_labels=0,final_Test_opened=False,expert_verified=False),ensure_ascii=False,indent=2))
print('Frozen partial2 normal2; Train-only; severe0 none; squat failure none')
