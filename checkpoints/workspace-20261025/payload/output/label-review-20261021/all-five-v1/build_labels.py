"""5종의 직접 검토 라벨을 통합하고 누락·중복·분할 조건을 검증한다.

입력은 이전 직접 검토 JSON과 이번에 사람이 확인 가능한 시간순 장면을 보고
Codex가 작성한 결정이다. 제공자 오류명이나 모델 예측을 점수로 변환하지 않는다.
출력 점수 단위는 명시된 관측 항목의 0~100 실험 점수이며 전문가 정답이 아니다.
"""
import hashlib
import html
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
NAMES = {'pushups':'푸시업','squats':'스쿼트','plank':'기본 팔꿈치 플랭크','ytw':'맨몸 전경 Y–T–W 레이즈','shoulder':'덤벨 숄더 프레스'}

# 직접 장면 검토 결과. 프로그램은 아래 결정을 판정하거나 생성하지 않는다.
SUPPLEMENT_DECISIONS = {
 1: ([1,2,1,2],[0,117],None,'Y에서 팔꿈치가 크게 굽혀진 채 전방으로 올린다. T 펼침은 충분하며 W는 부분 모양이다. 전경 몸통은 유지한다.'),
 2: ([1,2,1,1],[0,115],None,'Y 팔 굽힘과 부분 W가 보인다. T는 양옆으로 펼치며 전경 각도는 얕다.'),
 3: ([1,2,1,2],[0,115],None,'Y 팔이 굽혀지고 W는 부분 굽힘이다. T 펼침과 뚜렷한 전경 자세가 보이며 하단 복귀가 보인다.'),
 4: ([1,None,1,1],[0,129],'T의 양손이 화면 경계에서 잘려 도달 범위를 판단할 수 없음.','어두운 세로 영상이다. Y·W의 부분 팔 모양과 얕은 전경이 보이지만 T 손끝은 잘린다.'),
 5: ([None,1,None,2],[0,114],'Y·W의 손과 팔이 몸통·가구와 겹쳐 단계 도달을 구분하기 어려움.','T에서 팔꿈치가 굽혀진 부분 펼침이 보이며 전경은 유지된다. 손이 모여 있는 Y·W는 확정하지 않는다.'),
 6: ([1,1,1,1],[0,125],None,'얕은 전경에서 팔을 앞으로 짧게 굽혀 올리고 낮게 T를 펼친다. W도 부분적으로만 구분된다. 하단 복귀까지 보인다.'),
 7: ([1,2,1,0],[0,70],None,'몸통은 대체로 직립한 변형이다. Y는 굽혀 전방으로 올리고 T는 양옆으로 펼치며 W는 부분적으로 보인다. 목표인 전경 변형 기준에서 몸통 항목을 0으로 기록한다.'),
 8: ([None,2,1,2],[0,140],'첫 Y가 진행된 상태에서 영상이 시작되어 단일 전체 시도 시작이 확인되지 않음.','T와 부분 W, 전경 유지 및 복귀는 보이지만 첫 Y 시작은 영상 밖이다. 추측하여 총점을 부여하지 않는다.'),
 9: ([2,2,2,2],[0,31],None,'정면 첫 하단–상단–하단에서 양팔을 거의 펴 올린 뒤 함께 어깨 높이로 내린다. 큰 좌우 몸통 기울기는 보이지 않는다.'),
 10: ([None,2,2,2],[16,63],'최대 도달에서 손과 팔 끝이 상단 화면 밖으로 잘림.','하단 복귀와 동시 움직임은 보이나 위로 편 손이 화면 밖이므로 상단 도달은 판단 보류한다.'),
 11: ([None,2,2,2],[16,78],'앉은 프레스 변형이며 위쪽 손도 잘려 서서 하는 목표 총점 자료에서 제외.','의자에 앉은 프레스이다. 동시 움직임·하단 복귀와 정면 몸통은 관찰되나 서서 하는 프레스로 합치지 않는다.'),
 12: ([None,2,2,2],[0,63],'앉은 프레스 변형이고 상단 손이 잘림.','앉은 변형의 복귀·양팔 움직임을 항목 라벨로 보존한다. 위쪽 도달과 목표 변형 총점은 보류한다.'),
 13: ([None,2,2,2],[0,47],'상단 손과 덤벨이 화면 위 경계에서 잘림.','어깨 높이 복귀와 양팔 동시 움직임, 정면 몸통 유지는 보이지만 최대 도달은 보류한다.'),
 14: ([2,2,2,2],[0,55],None,'첫 하단–상단–하단에서 양팔을 거의 펴고 손까지 보인다. 복귀·양팔 동시 움직임·정면 몸통 유지가 확인된다.'),
 15: ([None,2,2,2],[0,39],'최대 도달 손과 팔 끝이 상단 밖으로 잘림.','복귀·동시 움직임·정면 몸통은 관측되나 위쪽 도달은 확정할 수 없다.'),
 16: ([None,2,2,2],[0,39],'최대 도달 손과 팔 끝이 상단 밖으로 잘림.','첫 동작의 하단 복귀와 동시 움직임은 보이나 상단 잘림 때문에 총점은 보류한다.')}

PLANK_DECISIONS = {
 1: ([2,2,2,2],None,'neutral','선택 3초 동안 전신이 보이고 몸통·골반의 큰 꺾임 없이 팔꿈치와 발끝 지지를 유지한다.'),
 2: ([0,2,2,2],'도입부의 반복 시범 가능성으로 9번과 묶고 9번만 대표 구간으로 사용.','hips_high','골반이 어깨–발목 선보다 크게 높다. 팔꿈치 전후 위치·편 무릎과 발끝 지지는 유지한다.'),
 3: ([0,2,None,2],'골반의 바닥 접촉·추가 지지 여부를 영상에서 확정하기 어려움.','hips_down','몸통이 크게 처지고 골반이 바닥에 가깝다. 접촉과 체중 지지 여부를 추측하지 않아 추가 지지 항목은 보류한다. 팔꿈치 위치와 구간의 자세 유지는 보인다.'),
 4: ([0,2,None,2],'3번과 같은 수행의 인접·편집 장면으로 중복 대표 제외.','hips_down','골반이 바닥에 가까운 채 몸통이 처진 같은 시범이 이어진다. 반복 표본 수를 늘리는 데 쓰지 않는다.'),
 5: ([None,2,1,1],'3초 안에 높은 골반→정렬→골반 바닥으로 바뀌어 하나의 안정된 품질 등급으로 묶을 수 없음.','transition','구간 내부 자세가 크게 달라져 몸통 단일 등급과 총점을 보류한다.'),
 6: ([0,2,None,None],'중간에 휴대전화 확대 화면으로 전환되어 연속 3초 전신 관찰이 안 됨.','hips_down','앞부분의 골반 처짐은 보이나 중간 편집 뒤에는 지지 유지 여부가 관측되지 않는다.'),
 7: ([0,2,None,1],'골반 처짐 시범은 3번과 같은 수행 묶음의 보충 근거로만 보존.','hips_down','골반을 바닥 가까이 내리고 휴대전화를 다룬다. 팔·손 위치 변경이 있어 지지 유지 항목은 부분 등급으로 기록한다.'),
 8: ([0,2,None,1],'7번과 같은 연속 시범이며 같은 골반 바닥 수행 묶음.','hips_down','골반 처짐 시범과 휴대전화 조작이 이어진다. 별도 독립 표본으로 세지 않는다.'),
 9: ([0,2,2,2],None,'hips_high','골반을 크게 들어 올린 채 무릎을 펴고 발끝과 팔꿈치 지지를 유지한다. 큰 골반 들림은 몸통 항목 미달로 판정한다.'),
 10: ([0,None,None,None],'전신·팔꿈치·발 지점이 잘리고 확대 구도가 바뀌어 전체 지지를 확인할 수 없음.','hips_high','몸통·골반의 큰 꺾임은 보이지만 화면 밖 관절·접점은 판단하지 않는다.'),
 11: ([1,2,2,2],None,'hips_moderately_high','골반이 어깨–발목 선보다 일부 높게 유지된다. 팔꿈치 전후 위치, 편 다리·발끝 지지와 구간 유지가 보인다. 등 윗부분의 둥근 모양 자체를 오류로 판정하지 않는다.'),
 12: ([None,2,None,None],'초반 확대 장면→전신 장면으로 편집되고 끝에서 머리가 잘려 같은 연속 3초 전신 기준을 충족하지 않음.','neutral_cut','일부 전신 장면은 대체로 정렬되지만 전체 선택 구간의 관측 누락을 만점으로 채우지 않는다.')}

PLANK_RULE = {'version':'project-plank-side-core-v2','expert_validated':False,'weights_are_project_choices':True,'definitions':{'plank':{'name_ko':NAMES['plank'],'unit':'측면 단일 장면의 약 3초 지지 구간','weights':[40,25,20,15],'items':[
 {'id':'trunk_line','name_ko':'몸통·골반 정렬','anchors':['큰 골반 들림·처짐','일부 골반 높이 이탈','큰 꺾임 없이 정렬']},
 {'id':'elbow_sagittal_position','name_ko':'측면 팔꿈치 전후 위치','anchors':['어깨 아래에서 큰 전후 이탈','일부 전후 이탈','대체로 어깨 아래']},
 {'id':'leg_and_body_support','name_ko':'편 다리·몸통 추가 지지 여부','anchors':['골반·무릎을 바닥에 둔 추가 지지','부분 무릎 굽힘 또는 지지 전환','무릎을 펴고 발끝 지지']},
 {'id':'support_continuity','name_ko':'선택 구간의 지지 지속','anchors':['명확한 지지 중단','지지 위치 변경','선택 구간의 지지 유지']}],
 'scope':'측면에서 보이는 핵심 네 항목만 평가. 팔꿈치 좌우 폭·목 세부·호흡·견갑·통증·안전은 포함하지 않음. 나쁜 정렬을 유지한 경우에도 지지 지속 자체와 정렬 등급은 구분함.'}},'compatibility':'기존 project-other-three-core-v1의 정면 팔꿈치 좌우 판정과 교환할 수 없음. 항목 ID와 기준 버전을 함께 사용.'}


def read(path):
    """UTF-8 JSON을 읽는다. 입력은 Path이며 반환은 JSON 객체이다. 파일 오류는 전파한다."""
    return json.loads(path.read_text())


def save(path, obj):
    """JSON 객체를 UTF-8 파일로 저장한다. 저장 실패 시 OSError를 전파한다."""
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2))


def build():
    """기존 판정과 신규 수동 판정을 통합한다.

    Returns:
        None: 라벨 JSON, 검증 결과, HTML 검토 보고서를 저장한다.
    Raises:
        AssertionError: 누락된 등급의 총점 생성, 중복 ID 또는 분할 충돌 발생 시.
    """
    rows=[]
    for f in ['review-v6/pushups_labels.json','review-v6/squats_labels.json','targeted-v3/labels.json','other-three-v1/labels.json']:
        for original in read(PARENT/f):
            r=dict(original);r['origin_label_file']=str(PARENT/f)
            r['hold_reason']=r.get('total_score_hold_reason') or r.get('hold_reason')
            r['eligible_for_item_training']=r.get('eligible_for_item_training', any(v is not None for v in r['criterion_levels']))
            r['label_group']=r.get('person',r['id'])
            rows.append(r)
    other=read(PARENT/'other-three-v1/rubric.json')
    for m in read(ROOT/'supplement_manifest.json'):
        n=m['review_number'];levels,frames,hold,note=SUPPLEMENT_DECISIONS[n];rule=other['definitions'][m['exercise']]
        eligible=not hold and None not in levels
        rows.append(dict(id=m['id'],exercise=m['exercise'],video_path=m['path'],video_sha256=m['video_sha256'],person=m['person'],label_group=m['person'],split=m['split'],labels=m['source_labels'],provider_label_source=m['label_source'],criterion_ids=[x['id'] for x in rule['items']],criterion_names_ko=[x['name_ko'] for x in rule['items']],criterion_weights=rule['weights'],criterion_levels=levels,score=sum(w*v/2 for w,v in zip(rule['weights'],levels)) if eligible else None,rubric_version=other['version'],segment=dict(start_frame=frames[0],end_frame=frames[1],fps=m['evidence']['fps']),score_scope=rule['exclusions'],note_ko=note,hold_reason=hold,eligible_for_item_training=True,eligible_for_total_score_training=eligible,label_source='codex_direct_temporal_frame_review',expert_validated=False,source_aware_review=True,evidence_images=[m['evidence']['path']],review_status='AI_provisional_experimental',review_method='24장 시간순 장면을 직접 확인. 연속 재생·전 프레임 검토는 아님.',observed_variant='seated_press' if n in [11,12] else 'target_variant',origin_label_file=str(ROOT/'direct_supplement_decisions.json')))
    source=ROOT/'sources/fitnessfaqs_plank.mp4';sha=hashlib.sha256(source.read_bytes()).hexdigest();rule=PLANK_RULE['definitions']['plank']
    for m in read(ROOT/'plank_segments.json'):
        n=m['review_number'];levels,hold,performance,note=PLANK_DECISIONS[n];eligible=not hold and None not in levels
        rows.append(dict(id=f'youtube_lismOShjHnA_{m["start_frame"]}_{m["end_frame"]}',exercise='plank',video_path=str(source),video_sha256=sha,person='youtube_lismOShjHnA_performer',label_group='youtube_lismOShjHnA_performer',performance_group='youtube_lismOShjHnA_'+performance,split='reference_only',labels=[],provider_label_source='none',source_url='https://www.youtube.com/watch?v=lismOShjHnA',source_title='FitnessFAQs: NEVER DO PLANKS LIKE THIS | 10 Most Common Mistakes',source_usage_status='public_video_training_rights_unverified',criterion_ids=[x['id'] for x in rule['items']],criterion_names_ko=[x['name_ko'] for x in rule['items']],criterion_weights=rule['weights'],criterion_levels=levels,score=sum(w*v/2 for w,v in zip(rule['weights'],levels)) if eligible else None,rubric_version=PLANK_RULE['version'],segment=dict(start_frame=m['start_frame'],end_frame=m['end_frame'],fps=m['fps']),score_scope=rule['scope'],note_ko=note,hold_reason=hold,eligible_for_item_training=True,eligible_for_total_score_training=False,label_quality_eligible_for_total=eligible,expert_validated=False,source_aware_review=True,label_source='codex_direct_temporal_frame_review',review_status='AI_provisional_reference',review_method='16장 시간순 장면 직접 검토. 연속 재생·전 프레임 검토 아님.',evidence_images=[m['evidence']],origin_label_file=str(ROOT/'direct_plank_decisions.json')))
    # 원래 출처 라벨, 항목 판정, 총점, 실제 학습 분할을 별도 보존한다.
    for r in rows:
        r['schema_version']='qevd-extended-posture-score-v2'
        r['exercise_name_ko']=NAMES[r['exercise']]
        r['whole_posture_score']=None
        r['label_quality_eligible_for_total']=r.get('label_quality_eligible_for_total',r.get('eligible_for_total_score_training',False))
        r['score_is_expert_ground_truth']=False
        r['score_target_origin']='Codex visual criterion grades, summed with project weights; not provider 0–100 scores'
        r['source_usage_status']=r.get('source_usage_status','preserve_source_dataset_terms; license_not_reaudited_in_this_labeling_step')
        r['fit_split_candidate']=bool(r['eligible_for_total_score_training'] and r['split']=='train')
    # 같은 원본 파일·참여자의 학습·검증 분할 혼입을 검사한다.
    assert len({r['id'] for r in rows})==len(rows)
    groups=defaultdict(set);hash_splits=defaultdict(set)
    for r in rows:
        lv=r['criterion_levels'];assert len(lv)==len(r['criterion_weights'])==4
        assert all(v is None or (type(v) is int and v in [0,1,2]) for v in lv)
        if None in lv:assert r['score'] is None
        if r['score'] is not None:assert abs(r['score']-sum(w*v/2 for w,v in zip(r['criterion_weights'],lv)))<1e-8
        if r['eligible_for_total_score_training']:assert r['score'] is not None and not r['hold_reason']
        assert Path(r['video_path']).is_file()
        if r['label_quality_eligible_for_total']:
            groups[r['label_group']].add(r['split']);hash_splits[r['video_sha256']].add(r['split'])
    collisions={k:sorted(v) for k,v in groups.items() if len(v)>1};hash_collisions={k:sorted(v) for k,v in hash_splits.items() if len(v)>1}
    assert not collisions,collisions
    assert not hash_collisions,hash_collisions
    summary=dict(time=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),all_five_have_numeric_label_examples=True,full_dataset_annotation_complete=False,all_five_ready_for_training=False,models_trained_this_step=False,active_web_model_changed=False,expert_validated=False,total_review_records=len(rows),new_review_records=28,exercises={})
    for ex in NAMES:
        rs=[r for r in rows if r['exercise']==ex]
        valid=[r for r in rs if r['label_quality_eligible_for_total']]
        summary['exercises'][ex]=dict(name_ko=NAMES[ex],review_records=len(rs),numeric_drafts=sum(r['score'] is not None for r in rs),quality_eligible_core_score_labels=len(valid),fit_split_candidates=sum(r['fit_split_candidate'] for r in rs),validation_candidates=sum(r['eligible_for_total_score_training'] and r['split'] in ['val','validation'] for r in rs),test_candidates=sum(r['eligible_for_total_score_training'] and r['split']=='test' for r in rs),reference_only_labels=sum(r['split']=='reference_only' for r in valid),score_label_people=len({r['person'] for r in valid}),item_label_records=sum(r['eligible_for_item_training'] for r in rs),score_distribution=dict(Counter(str(r['score']) for r in valid)))
        assert len(valid)>0
        save(ROOT/f'{ex}_labels.json',rs)
    save(ROOT/'direct_supplement_decisions.json',{str(n):dict(levels=d[0],frames=d[1],hold_reason=d[2],note_ko=d[3]) for n,d in SUPPLEMENT_DECISIONS.items()})
    save(ROOT/'direct_plank_decisions.json',{str(n):dict(levels=d[0],hold_reason=d[1],performance_group=d[2],note_ko=d[3]) for n,d in PLANK_DECISIONS.items()})
    save(ROOT/'rubrics.json',[read(PARENT/'scored-v1/rubric.json'),other,PLANK_RULE])
    save(ROOT/'labels.json',rows);save(ROOT/'summary.json',summary)
    save(ROOT/'review_queue.json',[dict(id=r['id'],exercise=r['exercise'],hold_reason=r['hold_reason'],source_usage_status=r['source_usage_status']) for r in rows if not r['eligible_for_total_score_training']])
    (ROOT/'labels.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
    save(ROOT/'verification.json',dict(passed=True,unique_ids=True,missing_criteria_never_filled=True,formula_checked=True,source_files_exist=True,eligible_group_split_collisions=collisions,eligible_byte_hash_split_collisions=hash_collisions,plank_reference_not_in_fit=True,active_model_unchanged=True))
    make_html(rows,summary)
    progress=read(PARENT/'progress.json');progress.update(stage='all_five_initial_score_label_corpus',time=summary['time'],current_priority=list(NAMES),pushups_squats_additional_work='included_in_all_five',all_five_report=str(ROOT/'index.html'),all_five_label_summary=summary['exercises'],training_running=False,all_five_ready_for_training=False,remaining=['plank_more_independent_people_and_reusable_training_source','ytw_shoulder_more_people_and_grade_balance','same_view_input_features','participant_grouped_model_evaluation'])
    save(PARENT/'progress.json',progress)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def make_html(rows, summary):
    """라벨 배열과 집계 결과를 한글 HTML로 저장한다. 반환값은 없으며 저장 오류는 전파한다."""
    esc=html.escape
    body=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>5종 자세 점수 라벨 검토</title><style>body{font:16px sans-serif;max-width:1150px;margin:35px auto;background:#16212a;color:#edf4f6;line-height:1.6}a{color:#70dfcb}td,th{border:1px solid #57717f;padding:9px}table{border-collapse:collapse;width:100%}details{border:1px solid #57717f;padding:12px;margin:12px 0}img{max-width:100%}video{max-width:720px;width:100%}</style><h1>5종 운동 자세 점수 모델용 초기 라벨</h1><p>푸시업·스쿼트의 재검토 라벨, QEVD 항목별 라벨, 나머지 세 운동의 초기 라벨과 이번 추가 28구간을 통합했습니다. 모든 종목에 수치 라벨 예시는 생겼지만, 보유 영상 전체의 라벨링이나 5종 학습 준비가 완료된 상태는 아닙니다.</p><p>직접 시간순 장면을 보고 항목을 0·1·2로 판정한 실험 라벨입니다. 점수는 그 판정에 프로젝트 배점을 적용한 학습 목표이며, 전문가 정답·모델 예측이 아닙니다. 화면 밖 항목은 null로 남깁니다. 모델 학습과 서비스 모델 교체는 이번 단계에서 하지 않았습니다.</p><table><tr><th>종목</th><th>검토 기록</th><th>품질 조건을 충족한 제한 점수 라벨</th><th>train 분할 후보</th><th>참고용</th><th>점수 라벨 참여자</th></tr>']
    for ex,s in summary['exercises'].items():body.append(f'<tr><td>{esc(s["name_ko"])}</td><td>{s["review_records"]}</td><td>{s["quality_eligible_core_score_labels"]}</td><td>{s["fit_split_candidates"]}</td><td>{s["reference_only_labels"]}</td><td>{s["score_label_people"]}</td></tr>')
    body+=['</table><p>검토 기록 수는 구간 단위이며 독립 영상·참여자 수와 다릅니다. train 외 검증·시험 자료는 그대로 유지하고 학습으로 옮기지 않습니다. 기존 점수가 숫자로 남아 있어도 재검토 보류인 자료는 학습 후보에서 제외합니다.</p><h2>종목별 해석과 남은 작업</h2><ul><li>푸시업: 측면 몸통·내려가기·팔 다시 펴기·지지 네 항목입니다. 손잡이 변형·불명확한 경계·가림은 별도 보류합니다.</li><li>스쿼트: 정면과 동기화된 측면을 함께 보는 기준입니다. 측면 하나만 입력한 모델의 정답으로 무조건 사용하지 않습니다.</li><li>플랭크: 공개 시범 영상 한 출처·한 수행자에서 정상, 큰 골반 들림, 일부 골반 들림의 대표 3구간을 라벨링했습니다. 도입·인접·반복 장면은 추가 독립 표본으로 세지 않았습니다. 공개 시청 가능 여부가 재사용 학습 권한을 뜻하지 않으므로 현재 참고용으로 분리했습니다. 여러 참여자의 재사용 가능한 측면 자료가 추가로 필요합니다.</li><li>Y–T–W: 양팔 도달과 전경 몸통을 판정하는 제한 점수입니다. 어두움·잘림·동작 시작 누락은 총점을 보류합니다.</li><li>덤벨 숄더 프레스: 정면 팔 도달·복귀·동시성·좌우 몸통만 평가합니다. 허리 젖힘 등 측면 문제를 포함한 전체 점수는 아닙니다. 앉은 변형은 서서 하는 목표 총점에서 제외합니다.</li></ul><p>다음 학습에는 종목별 참여자와 등급 분포를 늘리고, 같은 참여자를 학습·검증에 나누지 않는 평가가 필요합니다. 현재 숫자만으로 정확도를 말할 수 없습니다.</p><p><a href="labels.json">전체 라벨 JSON</a> · <a href="labels.jsonl">JSONL</a> · <a href="rubrics.json">한글 항목·배점·기준</a> · <a href="verification.json">구조 검증</a> · <a href="review_queue.json">보류 목록</a></p>']
    for ex in NAMES:
        body.append('<h2>'+esc(NAMES[ex])+' — 직접 검토 기록</h2>')
        for r in [x for x in rows if x['exercise']==ex]:
            display='총점 보류' if r['score'] is None else f'제한 점수 초안 {r["score"]:g}'
            body.append('<details><summary>'+esc(r['id']+' / '+display+' / '+r['split'])+'</summary><p>'+esc(r.get('note_ko',''))+'</p><p>항목: '+esc(str(list(zip(r['criterion_ids'],r['criterion_levels']))))+'</p><p>기준 버전: '+esc(r['rubric_version'])+' / 총점 보류 사유: '+esc(str(r['hold_reason']))+'</p><p>범위: '+esc(r.get('score_scope','기준 문서 참조'))+'</p>')
            paths=r.get('evidence_images') or r.get('audit_evidence') or [r.get('evidence_image')]
            for path in paths:
                if path and Path(path).exists():body.append('<img loading="lazy" src="'+esc(Path(path).as_uri())+'">')
            seg=r['segment'];src=Path(r['video_path']).as_uri()+f'#t={seg["start_frame"]/seg["fps"]},{seg["end_frame"]/seg["fps"]}'
            body.append('<video controls preload="none" src="'+esc(src)+'"></video></details>')
    body.append('</html>');(ROOT/'index.html').write_text('\n'.join(body))

if __name__=='__main__':
    build()
