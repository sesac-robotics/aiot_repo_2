"""Freeze nonexpert reviewed complete cycles, never infer failure from EOF."""
import json,sys,datetime
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/return-failure-expansion-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import assert_training_allowed,POLICY
source={r['review_id']:r for r in json.loads((O/'intake.frozen.json').read_text())['records']}
spec=[('N001',.5,3.,2),('N002',1.,3.27,2),('N003',1.25,4.14,2),('N004',.5,3.,2),('N005',.4,1.85,2),('N010',.5,2.37,1),('N010',2.37,4.37,1),('N012',.75,2.87,2),('N014',1.3,3.87,2)]
labels=[]
for n,a,b,g in spec:
 r=source[n];assert r['split']=='train';assert sha(r['path'])==r['source_sha256'];sheet=O/(n+'-dense.jpg');labels.append(dict(r,id=f"return_more_{n}_{round(a*1000):06}_{round(b*1000):06}",start=a,end=b,grades=[None,None,g,None],view='front',variant='visible_arm_extension_only',family='QEVD',whole_attempt_observed=True,train_eligible=True,annotation_status='8fps_chronological_visual_draft_nonexpert',expert_verified=False,review_sheet=str(sheet),review_sheet_sha256=sha(sheet),criterion='return_extension',evidence_note_ko=('하강 후 상승 끝에서도 양팔의 뚜렷한 굽힘이 남고 다음 하강으로 이어짐. 영상 종료나 휴식 전환이 아닌 완결 관측 시도. 최소 상승/무상승 0점과 구분해 부분 복귀 1점.' if g==1 else '하강 뒤 팔을 거의 편 상단이 다시 관찰됨. 제공자의 다른 오류 태그나 하체 접촉 여부를 이 항목의 복귀 오류로 복사하지 않음.'),overall_quality_grade=None,initial_correct_ready_required_for_absolute_extension_target=False,source_role_unchanged=True))
assert_training_allowed(labels)
held=[]
for n in ['N006','N008','N016']:
 r=source[n];held.append(dict(source_id=n,person=r['person'],split=r['split'],return_grade=None,train_eligible=False,reason='Near-extension boundary and projection cannot establish a confident grade1 versus2 from review; retain frozen role and hold.'))
payload=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),policy_sha256=sha(POLICY),reviewed=labels,held=held,grade0_added=0,no_grade0_failure_inferred_from_stopped_or_truncated_video=True,expert_verified=False,provider_tags_not_used_as_project_grades=True)
p=O/'labels.frozen.json';assert not p.exists();p.write_text(json.dumps(payload,ensure_ascii=False,indent=2));print('FROZEN_LABELS',len(labels),'normal_return',sum(r['grades'][2]==2 for r in labels),'partial_return',sum(r['grades'][2]==1 for r in labels),'severe0',0)
