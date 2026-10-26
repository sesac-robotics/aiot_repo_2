"""추가 직접 검토를 기존 5종 점수 라벨에 연결하고 품질 조건을 검증한다.

입력은 직접 결정 JSON·장면 근거·이전 라벨이며 출력은 QEVD 확장 JSON과
한글 검토 HTML이다. 항목 미관측은 null, 총점은 명시한 범위의 실험 0~100이다.
"""
import hashlib
import html
import json
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
NAMES={'pushups':'푸시업','squats':'스쿼트','plank':'기본 팔꿈치 플랭크','ytw':'맨몸 전경 Y–T–W 레이즈','shoulder':'덤벨 숄더 프레스'}
DVIDS_NOTICE='The appearance of U.S. Department of War (DoW) visual information does not imply or constitute DoW endorsement.'
EXTERNAL_DECISIONS={
 'airforce':{
 1:([2,2,2,2],None,'전신이 보이는 측면에서 몸통·골반에 큰 꺾임이 없고 팔꿈치가 대체로 어깨 아래에 있다. 무릎을 펴고 발끝 지지를 3초 유지한다.'),
 2:([2,2,2,2],'같은 수행자·같은 핵심 자세의 반복 근거로 대표 1번과 묶어 표본 수 증가에서 제외.','목 방향은 달라지지만 현재 네 핵심 항목에 목 세부는 포함하지 않는다. 같은 자세의 반복을 새 독립 표본으로 세지 않는다.'),
 3:([2,2,2,2],'같은 수행자·핵심 자세의 반복 근거로 대표 제외.','고개를 내린 구간이다. 현재 몸통·팔꿈치·다리·지지 기준은 유지되며 목 자체는 채점 범위 밖이다.'),
 4:([2,2,2,2],'같은 수행자·핵심 자세의 반복 근거로 대표 제외.','전신 지지와 정렬은 보이나 같은 시범 반복 구간이다.')},
 'marine':{
 1:([2,2,2,2],None,'앞쪽에서 흰 신발을 신고 머리를 묶은 수행자를 평가한다. 측면 전신의 큰 꺾임 없이 어깨 아래 팔꿈치, 편 무릎·발끝 지지를 유지한다. 뒤 수행자는 라벨 대상이 아니다.'),
 2:([None,None,None,None],'전신 장면에서 발 확대 장면으로 편집되어 3초 전체의 동일 항목 관측이 안 됨.','앞부분은 1번과 같은 정렬 시범이나 뒤에서 발만 확대된다. 누락된 몸통·팔꿈치를 채우지 않는다.'),
 3:([None,2,None,1],'다리 접점이 잘리고 골반 위치도 구간 안에 바뀌므로 전신 총점 보류.','어깨 아래 팔꿈치는 보이며 몸통·지지 위치가 바뀐다. 화면 밖 무릎·발 지지 상태는 판단하지 않는다.'),
 4:([None,2,None,None],'구도 전환·하체 잘림·골반 위치 변화로 단일 전신 구간이 아님.','일부 측면 팔꿈치 전후 위치만 확인한다. 편집과 변화를 단일 총점으로 합치지 않는다.'),
 5:([None,None,None,None],'측면 확대에서 정면 두 수행자로 구도가 바뀌며 전체 접점이 보이지 않음.','측면·정면 장면과 두 사람의 관측 항목을 혼합하지 않는다.'),
 6:([None,None,None,None],'정면 구도에서 전신 정렬·팔꿈치 전후 위치·하체 지지 확인 불가.','두 수행자가 앞에서 찍혔고 한쪽 팔꿈치가 옆으로 나와 있다. 측면 기준 점수로 변환하지 않는다.'),
 7:([None,None,None,None],'같은 수행자의 반복 시범이며 큰 자막이 몸통·팔을 가림.','앞쪽 수행자의 장면이나 자막이 겹쳐 핵심 관절 일부가 보이지 않는다.'),
 8:([None,None,None,None],'운동 수행 장면이 아닌 설명자 얼굴 장면.','운동 자세 라벨 대상이 아니다.')} }


def read(path):
    """UTF-8 JSON Path를 읽어 객체로 반환한다. 읽기·형식 오류는 전파한다."""
    return json.loads(path.read_text())


def save(path,obj):
    """객체를 UTF-8 JSON으로 저장한다. 반환값은 없고 저장 실패는 전파한다."""
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2))


def digest(path):
    """파일 Path의 SHA-256 문자열을 반환한다. 읽기 오류는 전파한다."""
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    """판정·집계·검토 화면을 생성하고 누락·분할 혼입을 검사한다.

    Returns:
        None: labels.json/jsonl, summary.json, verification.json, index.html 저장.
    Raises:
        AssertionError: 기준 불일치·원본 해시 변경·ID 중복·분할 혼입 발생 시.
    """
    old=read(PARENT/'all-five-v1/labels.json');rows=[dict(r) for r in old];new=[]
    rubrics=read(PARENT/'all-five-v1/rubrics.json');core=rubrics[0];other=rubrics[1];plank=rubrics[2]
    decisions=read(ROOT/'direct_decisions.json');manifest=read(ROOT/'candidate_manifest.json')
    for m in manifest:
        n=m['review_number'];d=decisions[str(n)];ex=m['exercise'];rule=(core if ex in ['pushups','squats'] else other)['definitions'][ex];version=(core if ex in ['pushups','squats'] else other)['version'];levels=d['levels'];eligible=not d['hold_reason'] and None not in levels
        new.append(dict(schema_version='qevd-extended-posture-score-v2',id=m['id'],exercise=ex,exercise_name_ko=NAMES[ex],video_path=m['path'],video_sha256=m['video_sha256'],person=m['person'],label_group=m['person'],split=m['split'],labels=m['source_labels'],provider_label_source=m['label_source'],criterion_ids=[i['id'] for i in rule['items']],criterion_names_ko=[i['name_ko'] for i in rule['items']],criterion_weights=rule['weights'],criterion_levels=levels,rubric_version=version,segment=dict(start_frame=d['frames'][0],end_frame=d['frames'][1],fps=m['evidence']['fps']),score=sum(w*v/2 for w,v in zip(rule['weights'],levels)) if eligible else None,whole_posture_score=None,score_scope=rule.get('scope',rule.get('exclusions')),unit='project_observable_core_0_to_100',note_ko=d['note_ko'],hold_reason=d['hold_reason'],eligible_for_item_training=any(v is not None for v in levels),eligible_for_total_score_training=eligible,label_quality_eligible_for_total=eligible,fit_split_candidate=eligible and m['split']=='train',source_aware_review=True,expert_validated=False,score_is_expert_ground_truth=False,label_source='codex_direct_temporal_frame_review',review_method='36장 시간순 장면 직접 검토. 연속 재생·전 프레임 검토 아님.',review_status='AI_provisional_experimental',source_usage_status='preserve_source_dataset_terms; license_not_reaudited_in_this_labeling_step',evidence_images=[m['evidence']['path']],origin_label_file=str(ROOT/'direct_decisions.json'),score_target_origin='Codex visual criterion grades, summed with project weights; not provider 0–100 scores',review_number=n))
    rule=plank['definitions']['plank']
    for source in ['airforce','marine']:
        segments=read(ROOT/f'{source}_segments.json');url='https://www.dvidshub.net/video/825280/forearm-plank' if source=='airforce' else 'https://www.dvidshub.net/video/871026/plank-instructional-video'
        for m in segments:
            n=m['review_number'];levels,hold,note=EXTERNAL_DECISIONS[source][n];eligible=not hold and None not in levels
            r=dict(schema_version='qevd-extended-posture-score-v2',id=f'dvids_{source}_{m["start_frame"]}_{m["end_frame"]}',exercise='plank',exercise_name_ko=NAMES['plank'],video_path=m['video_path'],video_sha256=digest(m['video_path']),person=f'dvids_{source}_target_performer',label_group=f'dvids_{source}_all_performers',performance_group=f'dvids_{source}_core_hold',split='train',labels=[],provider_label_source='none; instructional_text_not_converted_to_score',criterion_ids=[i['id'] for i in rule['items']],criterion_names_ko=[i['name_ko'] for i in rule['items']],criterion_weights=rule['weights'],criterion_levels=levels,rubric_version=plank['version'],segment={k:m[k] for k in ['start_frame','end_frame','fps']},score=sum(w*v/2 for w,v in zip(rule['weights'],levels)) if eligible else None,whole_posture_score=None,score_scope=rule['scope'],unit='project_observable_core_0_to_100',note_ko=note,hold_reason=hold,eligible_for_item_training=any(v is not None for v in levels) and not(source=='airforce' and n>1),eligible_for_total_score_training=eligible,label_quality_eligible_for_total=eligible,fit_split_candidate=eligible,source_aware_review=True,expert_validated=False,score_is_expert_ground_truth=False,label_source='codex_direct_temporal_frame_review',review_method='24장 시간순 확대 장면 직접 검토. 연속 재생·전 프레임 검토 아님.',review_status='AI_provisional_experimental',source_usage_status='US_government_official_video; DVIDS_noncommercial_notice_preserved; no_endorsement',source_url=url,source_terms_url='https://www.dvidshub.net/about/copyright',source_notice=DVIDS_NOTICE,evidence_images=[m['evidence']],origin_label_file=str(ROOT/'external_direct_decisions.json'),score_target_origin='Codex visual criterion grades, summed with project weights; not provider 0–100 scores',requires_subject_selection=source=='marine',subject_selection='가장 앞쪽 흰 신발·묶은 머리 수행자 한 명. 뒤 수행자를 합치지 않음.' if source=='marine' else '영상의 유일한 수행자',review_number=n)
            new.append(r)
    rows.extend(new);assert len({r['id'] for r in rows})==len(rows)
    # 실제 파일과 직접 판정이 연결되는지 검사한다. 학습 가능 여부와 항목 라벨 유무는 별도이다.
    cache={};groups=defaultdict(set);hashgroups=defaultdict(set)
    for r in rows:
        lv=r['criterion_levels'];assert len(lv)==len(r['criterion_weights'])==4
        assert sum(r['criterion_weights'])==100
        assert all(v is None or (type(v) is int and v in [0,1,2]) for v in lv)
        if None in lv:assert r['score'] is None
        if r['score'] is not None:assert abs(r['score']-sum(w*v/2 for w,v in zip(r['criterion_weights'],lv)))<1e-8
        if r['eligible_for_total_score_training']:assert r['score'] is not None and not r['hold_reason']
        if r['video_path'] not in cache:cache[r['video_path']]=digest(r['video_path'])
        assert cache[r['video_path']]==r['video_sha256'],r['id']
        assert r['segment']['fps']>0 and r['segment']['start_frame']<r['segment']['end_frame']
        if r['eligible_for_total_score_training']:
            groups[r['label_group']].add(r['split']);hashgroups[r['video_sha256']].add(r['split'])
    assert all(len(v)==1 for v in groups.values());assert all(len(v)==1 for v in hashgroups.values())
    inventory=read(ROOT/'inventory_before_review.json');screen=read(ROOT/'screen_manifest.json')
    for ex,s in inventory.items():
        s['direct_review_clips_now']=s['previous_direct_review_clips']+sum(r['exercise']==ex and str(r['provider_label_source']).startswith('QEVD') for r in new)
        s['screening_only_clips_now']=s['previous_screening_only']+(len(screen) if ex=='plank' else 0)
        s['unreviewed_and_unscreened_candidates']=s['local_candidate_clips']-s['direct_review_clips_now']-s['screening_only_clips_now']
        # 외부 프레스 데이터는 QEVD 후보에 포함하지 않는다.
        s['new_direct_review_qevd_clips']=sum(r['exercise']==ex and str(r['provider_label_source']).startswith('QEVD') for r in new)
        s['all_candidates_reviewed']=False
    screening_decisions=[dict(id=m['id'],person=m['person'],video_sha256=m['video_sha256'],evidence=m['evidence'],review_method='전체 영상의 20·50·80%에 해당하는 세 장면에서 구도만 직접 확인',eligible_for_total_score_review_from_sample=False,reason_ko='정면·사선 상체 중심 구도에서 측면 전신과 무릎·발 접점이 함께 확인되지 않음. 영상 전체를 불량이라고 판정한 것은 아님.',label_created=False) for m in screen]
    summary=dict(time=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),total_review_records=len(rows),new_direct_review_records=len(new),new_dataset_videos_reviewed=len(manifest),new_qevd_videos_reviewed=sum(str(r['provider_label_source']).startswith('QEVD') for r in new),new_external_plank_segment_reviews=12,new_screening_only_clips=len(screen),new_core_score_labels=sum(r['label_quality_eligible_for_total'] for r in new),full_dataset_annotation_complete=False,all_five_ready_for_training=False,models_trained_this_step=False,production_ready=False,expert_validated=False,exercises={})
    for ex in NAMES:
        rs=[r for r in rows if r['exercise']==ex];added=[r for r in new if r['exercise']==ex];valid=[r for r in rs if r['label_quality_eligible_for_total']]
        summary['exercises'][ex]=dict(name_ko=NAMES[ex],review_records=len(rs),quality_eligible_core_score_labels=len(valid),new_core_score_labels=sum(r['label_quality_eligible_for_total'] for r in added),new_item_label_records=sum(r['eligible_for_item_training'] for r in added),fit_split_candidates=sum(r['fit_split_candidate'] for r in rs),validation_candidates=sum(r['eligible_for_total_score_training'] and r['split'] in ['val','validation'] for r in rs),test_candidates=sum(r['eligible_for_total_score_training'] and r['split']=='test' for r in rs),reference_only_labels=sum(r['split']=='reference_only' for r in valid),score_label_people=len({r['person'] for r in valid}),score_distribution=dict(Counter(str(r['score']) for r in valid)))
        save(ROOT/f'{ex}_labels.json',rs)
    save(ROOT/'external_direct_decisions.json',EXTERNAL_DECISIONS);save(ROOT/'labels.json',rows);save(ROOT/'new_labels.json',new);save(ROOT/'summary.json',summary);save(ROOT/'rubrics.json',rubrics);save(ROOT/'qevd_inventory.json',inventory);save(ROOT/'screening_decisions.json',screening_decisions)
    save(ROOT/'review_queue.json',[dict(id=r['id'],exercise=r['exercise'],reason=r['hold_reason'],source_usage_status=r['source_usage_status']) for r in rows if not r['eligible_for_total_score_training']])
    (ROOT/'labels.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
    save(ROOT/'verification.json',dict(passed=True,unique_ids=True,actual_video_sha256_verified=True,source_files_verified=len(cache),missing_criteria_never_filled=True,formula_checked=True,source_person_split_leakage=False,source_byte_hash_split_leakage=False,old_labels_preserved=rows[:len(old)]==old,screening_not_counted_as_labels=True,external_repeated_holds_not_counted_as_independent_scores=True,expert_ground_truth_claimed=False,service_model_changed=False))
    save(ROOT/'dataset_manifest.json',dict(label_file_sha256=digest(ROOT/'labels.json'),rubric_file_sha256=digest(ROOT/'rubrics.json'),sources=[dict(path=k,sha256=v) for k,v in cache.items()]))
    report(summary,inventory,new)
    progress=read(PARENT/'progress.json');progress.update(stage='all_five_additional_direct_labeling_v2',time=summary['time'],all_five_report=str(ROOT/'index.html'),all_five_label_summary=summary['exercises'],training_running=False,all_five_ready_for_training=False,remaining=['plank_more_people_and_error_core_score_labels','ytw_shoulder_grade_balance_and_same_view_features','pushups_side_view_squats_synchronized_view_supplements','participant_grouped_model_evaluation'])
    save(PARENT/'progress.json',progress);print(json.dumps(summary,ensure_ascii=False,indent=2))


def report(summary,inventory,new):
    """집계·추가 판정·근거 이미지를 한글 HTML로 저장한다.

    Args:
        summary: 종목별 수량 집계. 단위는 검토 구간 수이다.
        inventory: QEVD 후보·검토·구도 확인 영상 수.
        new: 이번에 직접 검토한 라벨 객체 배열.
    Returns:
        None: index.html 파일을 저장한다.
    Raises:
        OSError: 파일을 저장하지 못하는 경우.
    """
    e=html.escape
    b=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>5종 추가 라벨링 — QEVD 검토 범위</title><style>body{font:16px sans-serif;max-width:1200px;margin:32px auto;background:#16212a;color:#e8f3f5;line-height:1.7}a{color:#70dfcb}table{border-collapse:collapse;width:100%}td,th{padding:9px;border:1px solid #536d7d}details{padding:12px;border:1px solid #536d7d;margin:10px 0}img{max-width:100%}video{max-width:800px;width:100%}</style><h1>5종 자세 점수 라벨링 — 추가 검토 결과</h1><p><strong>QEVD 관련 영상을 모두 검토한 상태는 아닙니다.</strong> 아래 후보 수는 적합 판정을 통과한 수가 아닙니다. 구도만 본 자료와 실제 항목을 판정한 자료를 구분합니다.</p><h2>QEVD에서 실제로 확인한 범위</h2><table><tr><th>종목</th><th>로컬 후보 영상</th><th>항목 직접 검토 영상</th><th>구도만 확인한 추가 영상</th><th>아직 미검토</th></tr>']
    for ex,s in inventory.items():b.append(f'<tr><td>{e(NAMES[ex])}</td><td>{s["local_candidate_clips"]}</td><td>{s["direct_review_clips_now"]}</td><td>{s["screening_only_clips_now"]}</td><td>{s["unreviewed_and_unscreened_candidates"]}</td></tr>')
    b+=['</table><p>숄더 프레스는 QEVD가 아닌 보유 다중 시점 데이터셋으로 진행합니다. 후보 영상의 제공자 정상·오류명은 표집·출처 정보이며, 0~100 점수로 자동 변환하지 않습니다.</p><h2>점수 라벨 증가</h2><table><tr><th>종목</th><th>누적 제한 점수 라벨</th><th>이번 추가</th><th>train 분할 후보</th><th>참고용</th><th>참여자</th></tr>']
    for ex,s in summary['exercises'].items():b.append(f'<tr><td>{e(s["name_ko"])}</td><td>{s["quality_eligible_core_score_labels"]}</td><td>+{s["new_core_score_labels"]}</td><td>{s["fit_split_candidates"]}</td><td>{s["reference_only_labels"]}</td><td>{s["score_label_people"]}</td></tr>')
    b+=['</table><p>이번에는 보유 영상 32개를 36장씩 시간순으로 검토하고, 외부 플랭크 12구간을 확대 24장씩 검토했습니다. 플랭크 새 참여자 60명은 3장씩 구도만 확인했으며 라벨 수에 넣지 않았습니다. 검토는 연속 재생·전 프레임 확인이나 전문가 검증과 구별됩니다.</p><h2>적합성 판정 결과</h2><ul><li>Y–T–W: 새 참여자의 완결 시도·팔 전체·몸통이 확인된 구간은 점수 라벨을 추가했습니다. 시작·복귀 누락, 손 잘림, 역광으로 관측하지 못한 항목은 null입니다.</li><li>숄더 프레스: 새 참여자 2명의 정면 영상 4개를 추가 확인했습니다. 최대 도달에서 손이 잘린 영상은 총점을 보류합니다. 정면 점수는 허리 젖힘까지 포함한 전체 자세 점수가 아닙니다.</li><li>QEVD 푸시업·스쿼트: 새 영상 각 8개에서 판정 가능한 항목을 라벨링했습니다. 정면에서 몸통·하체가 겹치거나 측면이 없어 총점 기준을 충족하지 못한 영상은 총점이 없습니다. 정상·오류명으로 빈 항목을 채우지 않았습니다.</li><li>QEVD 플랭크: 기존 16개 직접 검토와 120명 구도 확인에 더해 새 60명의 세 시점을 확인했습니다. 이번 표본에서는 측면 전신·지지 접점을 함께 확인하지 못했습니다. 전체 6,477개가 부적합하다는 뜻은 아닙니다.</li><li>외부 플랭크: DVIDS 공식 시범 두 영상에서 측면 전신 대표 정상 구간 1개씩을 추가했습니다. 같은 사람·같은 자세 반복 구간은 독립 표본 수로 늘리지 않았습니다. 다른 행사 영상도 확인했으나 구간 짧음·가림·다수 수행자 때문에 총점 자료로 채택하지 않았습니다.</li></ul><h2>아직 필요한 자료</h2><p>현재 플랭크의 새 학습 후보는 정상 점수뿐입니다. 여러 참여자의 정상·중간·오류 점수가 더 있어야 회귀 모델의 판별 능력을 평가할 수 있습니다. Y–T–W와 프레스도 참여자·등급 분포를 더 확보해야 합니다. 총점은 관측한 핵심 항목의 실험 라벨이며 전체 자세·안전·정확도 보증이 아닙니다. 이번 단계에서 모델을 재학습하거나 서비스 모델을 바꾸지는 않았습니다.</p><p>보유 데이터의 적합 후보를 계속 선별하면서, 구도·등급이 부족한 부분을 같은 기준의 외부 영상으로 보충하는 순서가 맞습니다. 외부 데이터를 섞을 때에는 촬영 방향, 운동 변형, 평가 항목, 참여자 분할과 출처 이용 조건을 함께 맞춥니다.</p><p><a href="labels.json">통합 JSON</a> · <a href="labels.jsonl">통합 JSONL</a> · <a href="new_labels.json">이번 추가 라벨</a> · <a href="rubrics.json">한글 항목·배점</a> · <a href="verification.json">검증 결과</a> · <a href="qevd_inventory.json">QEVD 검토 수량</a> · <a href="../all-five-v1/index.html">앞선 라벨 근거</a></p><h2>이번 직접 판정 근거</h2>']
    for r in new:
        title=f'{r["exercise_name_ko"]} / {r["id"]} / '+('총점 보류' if r['score'] is None else f'제한 점수 초안 {r["score"]:g}')
        b.append('<details><summary>'+e(title)+'</summary><p>'+e(r['note_ko'])+'</p><p>'+e(str(list(zip(r['criterion_names_ko'],r['criterion_levels']))))+'</p><p>보류: '+e(str(r['hold_reason']))+'</p>')
        if r.get('source_url'):b.append('<p><a href="'+e(r['source_url'])+'">원본 출처</a> · <a href="'+e(r['source_terms_url'])+'">이용 안내</a></p><p>'+e(DVIDS_NOTICE)+'</p>')
        for image in r['evidence_images']:b.append('<img loading="lazy" src="'+e(Path(image).as_uri())+'">')
        seg=r['segment'];src=Path(r['video_path']).as_uri()+f'#t={seg["start_frame"]/seg["fps"]},{seg["end_frame"]/seg["fps"]}'
        b.append('<video controls preload="none" src="'+e(src)+'"></video></details>')
    b+=['<h2>플랭크 구도 확인 자료</h2><p>다음 장면들은 점수 라벨이 아니며, 측면 전신 후보 선별을 위해 본 자료입니다.</p>']
    for n in range(1,6):b.append('<details><summary>구도 확인 '+str(n)+'</summary><img loading="lazy" src="overview_'+str(n).zfill(2)+'.jpg"></details>')
    b+=['<h2>외부 출처</h2><p><a href="https://www.dvidshub.net/video/825280/forearm-plank">Forearm Plank — 301st Fighter Wing / Randall Moose</a><br><a href="https://www.dvidshub.net/video/871026/plank-instructional-video">Plank Instructional Video — Marine Corps Base Quantico / Caden Phillips</a><br><a href="https://www.dvidshub.net/about/copyright">DVIDS 이용·비보증·비후원 안내</a></p><p>'+e(DVIDS_NOTICE)+'</p></html>']
    (ROOT/'index.html').write_text('\n'.join(b))

if __name__=='__main__':main()
