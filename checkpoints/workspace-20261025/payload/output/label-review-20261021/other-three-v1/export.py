"""세 운동의 직접 라벨과 관측 범위를 QEVD 확장 형식으로 내보낸다.

제공자 오류명은 보존만 하며 수치 점수로 바꾸지 않는다. 미관측 항목을 만점으로
채우지 않는다. 모든 점수는 프로젝트 실험 기준의 제한된 관측 항목 점수이다.
"""
import json
import html
from collections import Counter
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parent

RUBRIC={
 'version':'project-other-three-core-v1','expert_validated':False,'weights_are_project_choices':True,
 'level_meanings':{'0':'명확한 기준 미달이 직접 보임','1':'부분 미달이 직접 보임','2':'명시한 관측 항목 기준에 대체로 도달함','null':'가림·잘림·구도·등급 경계 때문에 판단 불가'},
 'formula':'관측 항목 모두 판정되고 단일 시도 구간이 확인된 경우에만 sum(weight * level / 2). 나머지는 null. 관측 항목 외의 전체 자세 점수는 항상 미검증.',
 'definitions':{
 'plank':{'name_ko':'기본 팔꿈치 플랭크','unit':'전신과 지지가 보이는 3초 유지 구간','required_view':'측면 전신·팔꿈치·무릎·발 접점','weights':[40,25,20,15],'items':[
 {'id':'trunk_line','name_ko':'몸통·골반 정렬','anchors':['큰 엉덩이 들림·처짐','일부 꺾임·구간 이탈','몸통·골반의 큰 꺾임 없이 유지']},
 {'id':'elbow_support_position','name_ko':'팔꿈치 지지 위치','anchors':['어깨와 팔꿈치의 큰 전후·좌우 이탈이 직접 보임','부분 위치 이탈','팔꿈치가 어깨 아래에 대체로 위치함']},
 {'id':'knee_trunk_support','name_ko':'무릎·몸통 추가 지지','anchors':['무릎·몸통을 바닥에 두고 지지함','일부 구간 추가 지지','관측 구간 팔꿈치·발끝 지지를 유지함']},
 {'id':'hold_continuity','name_ko':'관측 구간의 지지 유지','anchors':['확인되는 지지 중단','반복되는 지지 변경·큰 이탈','선택 3초에서 큰 지지 변경 없이 유지']}],
 'exclusions':'무릎 플랭크는 별도 변형으로 표시. 정면 팔만 보이는 자료에서 몸통 정렬·전신 유지 여부를 추측하지 않음. 유지 시간은 체력 점수가 아님.'},
 'ytw':{'name_ko':'맨몸 전경 Y–T–W 레이즈','unit':'한 번의 Y–T–W 시도와 원위치 복귀','required_view':'양팔 전체와 전경 몸통이 보이는 정면·사선','weights':[30,25,25,20],'items':[
 {'id':'y_reach','name_ko':'Y 단계 팔 도달','anchors':['Y 방향으로 뻗는 수행이 매우 작거나 명확히 미달','대각선 올림이 있으나 큰 팔 굽힘·부분 도달','양팔을 대각선 Y 방향으로 대체로 펴 올림']},
 {'id':'t_reach','name_ko':'T 단계 팔 펼침','anchors':['양옆 펼침이 매우 작음','부분 높이·큰 팔 굽힘·높이 차이','양팔을 어깨 선의 양옆 T 방향으로 대체로 펼침']},
 {'id':'w_shape','name_ko':'W 단계 팔 굽힘','anchors':['W 방향 굽힘 수행이 명확히 미달','부분 굽힘·팔 모양 차이','양 팔꿈치를 굽히고 손을 팔꿈치 위쪽으로 두는 W 모양이 구분됨']},
 {'id':'hinge_and_trunk_hold','name_ko':'전경 자세와 큰 몸통 흔들림','anchors':['대체로 직립이거나 몸통 큰 반동','얕은 전경 또는 일부 몸통 이탈','분명한 전경 자세를 큰 흔들림 없이 유지함']}],
 'exclusions':'정면 투영의 핵심 수행 범위. 척추 곡률·견갑 움직임·엄지 방향·통증·정확한 고관절 각도는 채점하지 않음. 다른 직립 변형을 임상적으로 잘못된 운동이라 부르지 않으며 목표 변형과 구분함.'},
 'shoulder':{'name_ko':'덤벨 숄더 프레스 — 정면 관측 핵심 항목','unit':'어깨 높이 하단 → 상단 → 하단의 단일 시도','required_view':'양팔과 머리 위 도달 범위가 보이는 정면','weights':[30,25,25,20],'items':[
 {'id':'overhead_reach','name_ko':'머리 위 팔 올림','anchors':['머리 위로 올림이 매우 작음','팔꿈치의 큰 굽힘·양팔 도달 차이','양팔을 머리 위로 거의 펴 올림. 강제 잠금은 요구하지 않음']},
 {'id':'lower_return','name_ko':'어깨 높이 하단 복귀','anchors':['하단으로 돌아오지 못함','부분 복귀','양팔이 어깨 높이 하단으로 대체로 복귀함']},
 {'id':'bilateral_timing','name_ko':'양팔 동시 움직임','anchors':['번갈아 올림·명확한 큰 시차','일부 높이·타이밍 차이','상승·하강에서 대체로 함께 움직임']},
 {'id':'frontal_trunk_hold','name_ko':'정면 몸통 좌우 유지','anchors':['큰 좌우 기울기·반동','부분 기울기·구간 이탈','큰 좌우 기울기 없이 유지함']}],
 'exclusions':'정면 핵심 항목의 제한 점수이며 전체 자세 점수가 아님. 허리 과신전·앞뒤 반동·견갑 기능·손목 세부는 이 시점에서 평가하지 않음. 손·팔 끝이 잘리면 최대 도달은 null. 정면 점수를 별도 측면 촬영에 전이하지 않음.'}},
 'sources':[
 {'url':'https://www.nasm.org/resource-center/exercise-library/plank','purpose':'팔꿈치 플랭크 정렬·지지 자세의 참고'},
 {'url':'https://www.evolutionphysicaltherapy.com/post/avoid-the-rock-climbing-hunchback-pt-3-strengthening/','purpose':'전경형 Y·T·W 팔 모양과 몸통 유지의 참고. 프로젝트 수치 배점의 검증 근거는 아님'},
 {'url':'https://www.acefitness.org/resources/everyone/exercise-library/186/seated-shoulder-press/','purpose':'머리 위 올림과 하강 동작의 참고. 본 실험은 서서 하는 프레스이며 정면 관측 범위만 라벨링'}],
 'temporal_limit':'장면 표본과 확대 근거를 직접 검토함. 연속 영상 재생·독립 전문가 검토를 완료했다고 표시하지 않음.',
 'clinical_limit':'점수는 통증·부상 위험·운동 효과를 평가하지 않음.'}


def export():
    """직접 결정만 합산하여 라벨·검토표·진행 기록을 저장한다.

    Returns:
        None: labels.json, summary.json, rubric.json, index.html을 출력한다.
    Raises:
        AssertionError: 라벨 누락·중복·범위 오류·미관측 총점이 발견되는 경우.
    """
    manifest=json.loads((ROOT/'candidate_manifest.json').read_text());decisions=json.loads((ROOT/'direct_decisions.json').read_text());labels=[];hashes=set()
    for r in manifest:
        n=r['review_number'];d=decisions[str(n)];rule=RUBRIC['definitions'][r['exercise']];levels=d['levels'];assert len(levels)==4 and all(v in [None,0,1,2] for v in levels);assert r['split']=='train'
        assert r['video_sha256'] not in hashes;hashes.add(r['video_sha256'])
        eligible=d['complete_unit'] and None not in levels and not d['hold_reason'];score=sum(w*v/2 for w,v in zip(rule['weights'],levels)) if eligible else None
        frames=d['unit_frames'] or [r['evidence']['review_start'],r['evidence']['review_end']];assert 0<=frames[0]<frames[1]<r['evidence']['total_frames']
        evidence=[r['evidence']['path']];detail=ROOT/f'detail_{n:03d}.jpg'
        if detail.exists():evidence.append(str(detail))
        row=dict(schema_version='qevd-extended-posture-score-v1',id=r['id'],original_id=r['original_id'],video_path=r['path'],video_sha256=r['video_sha256'],exercise=r['exercise'],person=r['person'],split=r['split'],labels=r['source_labels'],labels_descriptive=[d['note_ko']],provider_label_source=r['label_source'],rubric_version=RUBRIC['version'],criterion_ids=[x['id'] for x in rule['items']],criterion_names_ko=[x['name_ko'] for x in rule['items']],criterion_levels=levels,criterion_weights=rule['weights'],criterion_evidence_seconds=d['criterion_evidence_seconds'],score=score,whole_posture_score=None,unit='project_observable_core_0_to_100',score_scope=rule['exclusions'],required_view=rule['required_view'],segment=dict(start_frame=frames[0],end_frame=frames[1],fps=r['evidence']['fps']),completion_status='complete_single_unit_reviewed' if d['complete_unit'] else 'unit_or_visibility_unconfirmed',review_number=n,label_source='codex_direct_temporal_frame_review',reviewer='Codex',source_aware_review=True,expert_validated=False,review_status='AI_provisional_experimental',confidence=d['confidence'],eligible_for_total_score_training=eligible,eligible_for_item_training=any(v is not None for v in levels),eligible_for_experimental_training=eligible,hold_reason=d['hold_reason'],note_ko=d['note_ko'],evidence_images=evidence,observed_variant=d['observed_variant'],temporal_limit=RUBRIC['temporal_limit'])
        if None in levels:assert score is None
        labels.append(row)
    assert len(labels)==40
    summary=dict(time=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),priority='plank_ytw_shoulder_labeling_first',pushups_squats_additional_work='deferred',reviewed_video_candidates=40,additional_plank_screening_people=120,additional_plank_screening_basis='중간 한 장면의 촬영 방향 확인이며 점수 라벨링이 아님.',numeric_core_score_drafts=sum(r['score'] is not None for r in labels),models_trained_this_stage=False,production_ready=False,expert_validated=False,exercises={})
    summary['additional_plank_screening_basis']='중간 한 장면의 촬영 방향 확인이며 점수 라벨링이 아님.'
    for ex in RUBRIC['definitions']:
        rs=[r for r in labels if r['exercise']==ex];summary['exercises'][ex]=dict(reviewed=len(rs),numeric_core_score_drafts=sum(r['score'] is not None for r in rs),item_label_records=sum(r['eligible_for_item_training'] for r in rs),fully_unobservable=sum(not r['eligible_for_item_training'] for r in rs),held_total=sum(r['score'] is None for r in rs),people=len(set(r['person'] for r in rs)),score_counts=dict(Counter(str(r['score']) for r in rs if r['score'] is not None)))
    for name,obj in [('rubric.json',RUBRIC),('labels.json',labels),('summary.json',summary)]: (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
    for ex in RUBRIC['definitions']:(ROOT/f'{ex}_labels.json').write_text(json.dumps([r for r in labels if r['exercise']==ex],ensure_ascii=False,indent=2))
    body=['<!doctype html><meta charset="utf-8"><title>나머지 세 운동 초기 라벨 검토</title><style>body{font:16px sans-serif;max-width:1200px;margin:32px auto;background:#16212a;color:#e5eff4;line-height:1.6}a{color:#70daca}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #536873}details{padding:12px;border:1px solid #536873;margin:10px 0}img{max-width:100%}video{max-width:720px;max-height:500px}</style><h1>플랭크·Y–T–W·덤벨 숄더 프레스 — 초기 점수 라벨</h1><p>현재 우선순위는 세 종목의 직접 라벨링입니다. 푸시업·스쿼트의 추가 라벨링·재학습·Rule Base 비교는 뒤로 두었습니다.</p><p>총점은 명시한 관측 핵심 항목의 실험 점수입니다. 전체 자세·안전·전문가 정확도를 의미하지 않습니다. 점수를 붙이지 못한 영상도 그대로 보류 사유를 남겼습니다. 이번 단계에서는 모델을 학습하지 않았습니다.</p><table><tr><th>종목</th><th>대표 영상 검토</th><th>항목 라벨이 있는 영상</th><th>제한된 총점 초안</th><th>총점 보류</th></tr>']
    for ex,r in summary['exercises'].items():body.append(f"<tr><td>{RUBRIC['definitions'][ex]['name_ko']}</td><td>{r['reviewed']}</td><td>{r['item_label_records']}</td><td>{r['numeric_core_score_drafts']}</td><td>{r['held_total']}</td></tr>")
    body.append('</table><h2>현재 확인한 자료 문제</h2><p>플랭크는 대표 16개와 추가 참여자 120명의 중간 장면을 확인했지만 측면 전신·지지 접점을 판정할 구도를 찾지 못했습니다. QEVD 전체 6,477개가 모두 불가능하다는 결론은 아닙니다. 추가 선별 또는 측면 자료가 필요합니다. 다른 데이터셋의 abs 측면 표본 6개도 확인했으나 서서 수행하는 중량 동작으로 플랭크를 대체할 수 없었습니다.</p><p>Y–T–W는 Y·T·W 단계 구간과 팔 전체의 가림을 구분합니다. 숄더 프레스는 정면 핵심 항목을 라벨링했습니다. 정면 점수가 높아도 옆에서만 보이는 허리 젖힘까지 정상이라는 뜻은 아닙니다.</p><h2>항목과 배점</h2>')
    for ex,r in RUBRIC['definitions'].items():
        body.append('<details><summary>'+r['name_ko']+'</summary><p>'+html.escape(r['unit']+' / '+r['required_view'])+'</p><ul>')
        for item,w in zip(r['items'],r['weights']):body.append('<li>'+html.escape(item['name_ko'])+f' {w}점 — '+html.escape(' / '.join(item['anchors']))+'</li>')
        body.append('</ul><p>'+html.escape(r['exclusions'])+'</p></details>')
    body.append('<h2>직접 검토 근거</h2>')
    for r in labels:
        title=f"{r['review_number']:03d} {RUBRIC['definitions'][r['exercise']]['name_ko']} — "+(f"제한 점수 초안 {r['score']:g}" if r['score'] is not None else '총점 보류')
        body.append('<details><summary>'+html.escape(title)+'</summary><p>'+html.escape(r['note_ko'])+'</p><p>'+html.escape(str(dict(zip(r['criterion_names_ko'],r['criterion_levels']))))+'</p>')
        for path in r['evidence_images']:body.append('<a href="'+html.escape(Path(path).as_uri())+'">근거 확대</a><img src="'+html.escape(Path(path).as_uri())+'">')
        seg=r['segment'];clip=f"#t={seg['start_frame']/seg['fps']},{seg['end_frame']/seg['fps']}";body.append('<video controls preload="none" src="'+html.escape(Path(r['video_path']).as_uri()+clip)+'"></video></details>')
    body.append('<h2>수행 기준 참고</h2>')
    for src in RUBRIC['sources']:body.append('<p>'+html.escape(src['purpose'])+' — <a href="'+src['url']+'">원문</a></p>')
    body.append('<h2>다음 작업</h2><ol><li>Y–T–W와 프레스의 경계가 확인된 후보에서 관절 추출 가능률을 확인하고 새 참여자·정상·중간·명확한 미달을 보충합니다.</li><li>플랭크는 측면 전신 자료를 우선 선별합니다. 정면 자료의 제공자 오류명을 임의의 0~100 정답으로 바꾸지 않습니다.</li><li>숄더 프레스의 앞뒤 몸통 평가에는 별도 측면 자료가 필요합니다. 촬영 시점이 다른 동작을 같은 반복으로 합치지 않습니다.</li><li>세 종목의 라벨 구성이 마련된 뒤 모델 학습·Rule Base 비교를 진행합니다. 현재 세 종목 모델 완료 상태가 아닙니다.</li></ol>')
    (ROOT/'index.html').write_text('\n'.join(body));print(json.dumps(summary,ensure_ascii=False,indent=2))
    progress_path=ROOT.parent/'progress.json';progress=json.loads(progress_path.read_text());progress.update(stage='other_three_initial_direct_labeling',time=summary['time'],current_priority=['plank','ytw','shoulder'],pushups_squats_additional_work='deferred',other_three_report=str(ROOT/'index.html'),other_three_reviewed=40,other_three_numeric_core_drafts=summary['numeric_core_score_drafts'],training_running=False);progress_path.write_text(json.dumps(progress,ensure_ascii=False,indent=2))

if __name__=='__main__':export()
