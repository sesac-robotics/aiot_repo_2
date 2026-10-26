"""Codex의 영상 장면 검토를 항목별 점수 초안과 QEVD 확장 형식으로 기록한다.

스크립트는 영상을 판단하거나 오류를 생성하지 않는다. 아래 ASSESSMENTS는 Codex가
실제 확인한 시간 표시 장면의 수기 판단이다. 집계 수식과 보고서 출력만 자동화한다.
점수는 교육 프로젝트 정의이며 전문가 정답·임상 평가·학습 모델 예측이 아니다.
"""
import collections
import hashlib
import html
import json
from pathlib import Path

ROOT=Path('/home/jayden/SeSACPAI-Mid/output/label-review-20261021')
OUT=ROOT/'scored-v1'
# 첫 배열: 항목별 0(미충족), 1(부분 충족), 2(충족), None(관측 불가).
# 시간은 푸시업 원본 영상 기준 초, 스쿼트는 반복 시작을 0초로 둔 상대 초이다.
ASSESSMENTS={
1:([2,2,2,2],[1.67,2.33,3.0],['상승 시 어깨–엉덩이–발목의 큰 꺾임이 보이지 않는다.','2.33초에 상완이 바닥에 가까운 방향까지 내려간다.','1.67·3.00초에 팔을 편 상단 자세로 돌아온다.','선택한 구간에서 손과 발로 지지하고 무릎을 바닥에 대지 않는다.'],[]),
2:([2,2,2,2],[.67,1.33,1.67],['몸통과 다리가 함께 높이를 바꾼다.','1.33초에 팔이 충분히 굽혀진 하단 자세가 보인다.','0.67·1.67초에 팔을 편 자세가 보인다.','손·발 지지가 유지된다.'],[]),
3:([2,2,2,2],[.67,1.67,2.33],['선택 구간에서 엉덩이의 뚜렷한 들림·처짐이 보이지 않는다.','1.67초의 팔 굽힘과 흉부 하강이 확인된다.','0.67·2.33초의 팔 펴기가 확인된다.','선택 구간의 무릎은 바닥에 닿지 않는다.'],[]),
4:([2,2,2,2],[1.0,1.67,2.33],['몸통·다리 정렬이 대체로 유지된다.','1.67초에 가슴을 낮춘 자세가 보인다.','1.00·2.33초에 팔을 편 상단 자세가 보인다.','선택 구간의 손·발 지지가 유지된다.'],[]),
5:([2,2,2,2],[6.67,7.33,7.67],['선택한 가림이 적은 구간에서 어깨–엉덩이–발목 정렬이 유지된다.','7.33초에 팔 굽힘과 흉부 하강이 보인다.','6.67·7.67초 상단 자세의 팔 펴기가 보인다.','선택 구간의 무릎은 바닥에 닿지 않는다.'],[]),
6:([0,2,2,2],[1.33,2.33,3.0],['상단과 하단에서 엉덩이가 어깨–발목 선보다 뚜렷하게 높다.','2.33초 팔 굽힘과 흉부 하강이 있다.','1.33·3.00초 팔 펴기가 보인다.','손·발 지지가 유지된다.'],['hips_high']),
7:([2,2,1,2],[1.33,2.33,3.33],['선택 구간 몸통·다리에 뚜렷한 큰 꺾임이 보이지 않는다.','2.33초에 팔을 굽혀 충분히 낮아진다.','1.33·3.33초 상단에서도 팔꿈치 굽힘이 남아 보인다.','선택 구간의 손·발 지지가 유지된다.'],['incomplete_extension']),
8:([0,2,2,2],[3.67,5.0,5.67],['상단에서 어깨–엉덩이–발목 선이 엉덩이에서 뚜렷하게 꺾인다.','5.00초에 흉부가 낮아진다.','3.67·5.67초에 팔이 펴진다.','선택 구간 손·발 지지가 유지된다.'],['hips_high']),
9:([None,2,2,2],[1.33,2.33,3.0],['하강 구간에서 몸통·골반 선이 상단보다 아래로 꺾여 보인다. 경미한 처짐 초안이다.','2.33초에 흉부 하강과 팔 굽힘이 보인다.','1.33·3.00초 팔을 편 자세로 돌아온다.','선택 구간 손·발 지지가 유지된다.'],['possible_hips_sag']),
10:([2,0,2,2],[1.0,1.33,2.33],['어깨–엉덩이–발목 선에 뚜렷한 큰 꺾임이 보이지 않는다.','1.33초 전후 하강에서도 팔 굽힘·흉부 하강 범위가 작다.','1.00·2.33초 상단에서 팔 펴기가 보인다.','선택 구간 손·발 지지가 유지된다.'],['shallow']),
11:([1,2,2,2],[0,1.33,2.97],['cam3 1.33초에서 허벅지가 수평에 이르기 전 하강이 끝나 보인다.','2.97초 무릎·엉덩이를 펴고 서기로 돌아온다.','cam2에서 뚜렷한 무릎의 안쪽 붕괴가 보이지 않고 발이 유지된다.','cam2에서 뚜렷한 좌우 몸통 기울기가 보이지 않는다.'],['limited_depth_project_criterion']),
12:([2,2,2,2],[0,.67,1.5],['cam1 0.67초에서 허벅지가 수평에 가까운 높이까지 내려간다.','1.50초 다시 선다.','cam3에서 무릎·발의 정렬이 대체로 유지된다.','cam3에서 좌우 몸통 균형이 대체로 유지된다.'],[]),
13:([1,2,2,2],[0,.93,2.07],['cam3 0.93초에서 허벅지가 수평보다 위에 있어 보인다.','2.07초 다시 선다.','cam2에서 뚜렷한 안쪽 무릎 붕괴·발 이동이 보이지 않는다.','cam2에서 좌우 몸통 기울기가 뚜렷하지 않다.'],['limited_depth_project_criterion']),
14:([1,2,2,2],[0,1.0,2.27],['cam3 1.00초에서 허벅지가 수평보다 위에 있어 보인다.','2.27초 다시 선다.','cam1에서 뚜렷한 무릎의 안쪽 붕괴가 보이지 않는다.','cam1에서 좌우 몸통 기울기가 뚜렷하지 않다.'],['limited_depth_project_criterion']),
15:([1,2,2,2],[0,1.6,3.57],['cam3 1.60초에서 허벅지가 수평보다 위에 있어 보인다.','3.57초 다시 선다.','cam2에서 뚜렷한 안쪽 무릎 붕괴가 보이지 않는다.','cam2에서 뚜렷한 좌우 몸통 기울기가 보이지 않는다.'],['limited_depth_project_criterion']),
16:([2,2,0,2],[0,.9,2.0],['cam3 0.90초에 허벅지가 수평에 가까워진다.','2.00초 다시 선다.','cam2 0.70–1.07초에서 두 무릎이 발의 방향보다 안쪽으로 뚜렷하게 모인다.','cam2에서 뚜렷한 좌우 몸통 기울기는 보이지 않는다.'],['knees_in']),
17:([1,2,2,0],[0,1.2,2.67],['cam3 하단에서 허벅지가 수평보다 위에 있어 보인다.','2.67초 다시 선다.','cam2에서 뚜렷한 무릎 붕괴는 확인되지 않는다.','cam2 0.70–1.43초에서 어깨선·몸통이 한쪽으로 기울어진다.'],['trunk_lateral_tilt','limited_depth_project_criterion']),
18:([1,2,2,None],[0,1.23,2.73],['cam1 1.23초 하단에서 허벅지가 수평보다 위에 있어 보인다.','2.73초 다시 선다.','cam3에서 뚜렷한 안쪽 무릎 붕괴가 보이지 않는다.','cam3 0.97–1.73초에 몸통의 한쪽 기울기가 보인다. 정도는 재검토한다.'],['possible_trunk_lateral_tilt','limited_depth_project_criterion']),
19:([0,2,2,2],[0,.63,1.4],['cam1 0.37–0.73초에서 하강 범위가 작고 허벅지가 수평에서 크게 떨어져 있다.','1.40초 다시 선다.','cam3에서 무릎·발의 뚜렷한 붕괴가 보이지 않는다.','cam3에서 뚜렷한 좌우 몸통 기울기가 보이지 않는다.'],['shallow']),
20:([2,2,None,2],[0,.83,1.53],['cam1 0.83초에서 허벅지가 수평에 가까워진다.','1.53초 다시 선다.','원본 오류는 무릎의 전방 이동이다. 전방 이동만으로 이 항목을 감점하지 않으며 발 접지의 미세 변화는 이 자료로 확정하지 않는다.','cam3에서 뚜렷한 좌우 몸통 기울기가 보이지 않는다.'],['provider_definition_requires_resolution'])}


def build():
    """항목별 판단을 검증하고 JSON·CSV·HTML로 출력한다.

    Returns:
        None: 20개 라벨 초안과 채점표, 검증 요약을 출력한다.
    Raises:
        AssertionError: 항목 수·범위·근거·영상 구간 대응이 맞지 않는 경우.
        OSError: 자료 파일 읽기·쓰기 실패.
    """
    manifest=json.loads((ROOT/'manifest.json').read_text());views={r['review_id']:r for r in json.loads((ROOT/'multiview_manifest.json').read_text())}
    definitions={
      'pushups':dict(scope='측면에서 관측 가능한 핵심 수행 품질. 좌우 대칭·팔 벌림·머리 정렬·속도는 포함하지 않음.',weights=[40,30,20,10],items=[
       dict(id='trunk_line',name_ko='몸통·골반 정렬',anchors=['뚜렷한 엉덩이 들림·처짐','경미한 꺾임 또는 일부 구간 이탈','선택 구간에서 큰 꺾임 없이 정렬 유지']),
       dict(id='descent_range',name_ko='팔 굽힘·하강 범위',anchors=['움직임이 작고 하단에서도 팔이 크게 굽혀지지 않음','하강은 있으나 기준 자세에 충분히 이르지 않음','가슴이 낮아지고 상완이 바닥에 가까운 방향까지 내려감']),
       dict(id='return_extension',name_ko='상단 팔 펴기',anchors=['팔을 편 상단으로 돌아오지 못함','상단에서도 팔꿈치 굽힘이 일부 남음','상단에서 팔을 거의 편 자세로 돌아옴. 강제 잠금은 요구하지 않음']),
       dict(id='support',name_ko='손·발 지지',anchors=['무릎·몸통을 바닥에 대고 지지함','중간에 지지 방식이 바뀜','선택 구간 손·발 지지 유지'])]),
      'squats':dict(scope='동기화된 세 카메라로 관측한 반복의 핵심 수행 품질. 재활 개인 목표·통증·속도 평가는 아님.',weights=[30,20,30,20],items=[
       dict(id='descent_range',name_ko='하강 범위',anchors=['무릎·엉덩이를 조금만 굽힌 얕은 하강','분명히 내려가지만 허벅지가 수평보다 위에 있음','허벅지가 수평에 가까워지는 하강']),
       dict(id='return_extension',name_ko='다시 일어서기',anchors=['서기 자세로 복귀하지 못함','무릎·엉덩이 굽힘이 남아 부분 복귀','선택 반복에서 원래의 서기 자세로 복귀']),
       dict(id='knee_foot',name_ko='무릎·발 정렬과 접지',anchors=['뚜렷한 안쪽 무릎 붕괴·뒤꿈치 들림·발 지지 이탈','부분적 정렬 이탈','앞뒤·정면 근거에서 큰 정렬 이탈 없이 접지 유지']),
       dict(id='trunk_balance',name_ko='몸통 좌우 균형',anchors=['뚜렷한 한쪽 기울기·비대칭','경미한 한쪽 기울기','정면 근거에서 큰 좌우 기울기 없이 수행'])])}
    rubric=dict(version='project-core-form-v1',expert_validated=False,weights_are_project_choices=True,definitions=definitions,
      formula='sum(weight * level / 2); 모든 항목이 관측 가능할 때만 산출. null이면 총점 null. 관찰 등급은 Codex가 정하고 스크립트는 합산만 함.',
      interpretation='100은 이 제한된 항목의 만점이며 완벽한 자세·부상 없음·전문가 100점을 뜻하지 않음. 소수점은 배점 합산 결과이며 그 정밀도로 검증된 점수가 아님.',
      version_change='이전 제안의 측면에서 볼 수 없는 좌우 균형을 푸시업 총점에서 제외하고 손·발 지지로 변경. 두 버전 점수는 혼합 금지.',
      sources=[dict(title='NASM Push-Up',url='https://www.nasm.org/resource-center/exercise-library/push-up'),dict(title='ACE Bodyweight Squat',url='https://www.acefitness.org/resources/everyone/exercise-library/135/bodyweight-squat/')],
      temporal_limit='장면 표본의 상단–하단–상단 전환을 확인함. 연속 재생 또는 모든 프레임을 직접 검토한 것은 아님. 순간 반동·통증·정확한 관절각은 평가하지 않음.',
      provider_policy='good/bad와 REHAB26 1/5는 별도 제공자 라벨. 연속 점수로 직접 변환하지 않음. MultiView 다른 시점의 점수를 주 영상에 전이하지 않음.',
      test_policy='제공자 라벨을 본 탐색적 채점이다. 독립 정확도 검증으로 쓰지 않음. 참여자 및 수행 그룹 분리 유지.')
    labels=[]
    for r in manifest:
        number=r['review_id'];levels,times,reasons,errors=ASSESSMENTS[number];d=definitions[r['exercise']]
        assert len(levels)==len(reasons)==len(d['items'])==4
        assert len(times)==3 and times[0]<times[1]<times[2]
        assert all(v in [None,0,1,2] for v in levels)
        items=[dict(criterion_id=item['id'],name_ko=item['name_ko'],level=level,weight=weight,evidence_ko=reason,evidence_times_seconds=times) for item,level,weight,reason in zip(d['items'],levels,d['weights'],reasons)]
        score=None if None in levels else sum(w*v/2 for w,v in zip(d['weights'],levels))
        if r['exercise']=='pushups':
            segment=dict(start_seconds=times[0],end_seconds=times[-1],time_basis='original_video_seconds',selection='상단–하단–상단 전환의 수동 선택 구간',boundary_precision_seconds=1/3)
        else:segment=dict(start_frame=r['start_frame'],end_frame=r['end_frame'],fps=30,time_basis='evidence_seconds_relative_to_segment',selection='제공자 반복 구간')
        limitations=['미검토 자동 점수가 아닌 Codex 직접 장면 검토 초안이지만 전문가 정답은 아님.','샘플 장면 사이의 순간 자세는 관측하지 못할 수 있음.']
        if r['exercise']=='pushups':limitations+=['다른 카메라는 동기화가 확인되지 않아 별도 수행으로 취급. 총점은 주 측면 영상 구간만 사용.','이 총점은 좌우 대칭·팔 벌림·머리 정렬을 평가하지 않음.']
        if number in [9,18]:limitations+=['경미한 정렬 이탈의 정도가 애매해 독립 재검토 전 학습 후보로 승인하지 않음.']
        if number in [11,13,14,15]:limitations+=['제공자는 재활 과제 정상으로 분류했지만 프로젝트의 근력 운동 하강 기준은 다름. 임상적으로 잘못됐다는 뜻이 아님.']
        old_path=ROOT.parent/'overnight-20261020/score_labels'/f'{r["id"]}.json';old=json.loads(old_path.read_text()) if old_path.exists() else {}
        image=Path(views[number]['image']);imagehash=hashlib.sha256(image.read_bytes()).hexdigest()
        labels.append(dict(video_path=r['path'],labels=r['source_labels'],labels_descriptive=reasons,split=r['split'],
          schema_version='qevd-extended-posture-score-v1',id=r['id'],exercise=r['exercise'],person=r['person'],
          label_source='codex_direct_sampled_frame_review',reviewer='Codex',review_status='AI_reviewed_project_rubric_draft',rubric_version=rubric['version'],
          provider=dict(dataset=r['label_source'],labels=r['source_labels'],reference=r['reference']),segment=segment,
          items=items,score=score,score_unit='project_core_form_0_to_100',score_scope=d['scope'],observed_errors=errors,
          source_aware_review=True,expert_validated=False,eligible_for_training=False,
          experimental_candidate=score is not None and number not in [9,18],
          training_hold_reason='첫 표본의 기준·반복 일관성 검증 전. 기존 자동 학습 경로에 투입하지 않음.',
          evidence=dict(multiview_image=str(image),sha256=imagehash,dense_image=str(ROOT/'dense'/f'{number:02d}.jpg') if number<=10 else None,
              sample_rate_fps=3 if number<=10 else None,sampled_points_per_camera=12,full_video_playback_reviewed=False,
              cameras=views[number]['views'],synchronization=views[number]['synchronization']),
          previous_auto_score=old.get('score'),limitations_ko=limitations))
    assert len(labels)==20 and len({r['id'] for r in labels})==20
    (OUT/'rubric.json').write_text(json.dumps(rubric,ensure_ascii=False,indent=2))
    (OUT/'labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2))
    summary=dict(reviewed_records=20,scored_drafts=sum(r['score'] is not None for r in labels),
        score_withheld=sum(r['score'] is None for r in labels),experimental_candidates=sum(r['experimental_candidate'] for r in labels),training_approved=0,
        score_distribution={ex:dict(collections.Counter(str(r['score']) for r in labels if r['exercise']==ex)) for ex in definitions},
        expert_accuracy=None,learning_curve=None,old_119_labels_blocked=True,
        next_steps=['경미한 처짐·기울기와 하강 경계의 반복 재검토','대표 반복 구간을 추가 확인해 동일 기준의 일관성 검사','훈련·검증·시험 참여자별 표본 확보 후 학습 곡선 측정'])
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    import csv
    with (OUT/'labels.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','운동','점수_초안','검토_상태','학습_승인','출처_라벨'])
        for r in labels:w.writerow([r['id'],r['exercise'],r['score'],r['review_status'],r['eligible_for_training'],';'.join(r['labels'])])
    cards=[]
    for n,r in enumerate(labels,1):
        rows=''.join(f'<tr><td>{html.escape(x["name_ko"])}</td><td>{x["level"] if x["level"] is not None else "판단 보류"} / 2</td><td>{html.escape(x["evidence_ko"])}</td></tr>' for x in r['items'])
        score='보류' if r['score'] is None else f'{r["score"]:g}점'
        cards.append(f'<article><h2>{n:02d}. {html.escape(r["id"])}</h2><p class="score">프로젝트 기준 점수 초안: {score}</p><p>원본 라벨: {html.escape(str(r["labels"]))} · 기존 자동 점수: {r["previous_auto_score"]}</p><p>평가 구간: {html.escape(str(r["segment"]))}</p><table><tr><th>항목</th><th>등급</th><th>관찰 근거</th></tr>{rows}</table><p>{html.escape(" / ".join(r["limitations_ko"]))}</p><details><summary>세 시점 비교 장면</summary><img src="../multiview/{n:02d}.jpg"></details>'+ (f'<details><summary>측면 3fps 검토 장면</summary><img src="../dense/{n:02d}.jpg"></details>' if n<=10 else '')+'</article>')
    intro='<h1>푸시업·스쿼트 자세 점수 라벨 초안 — 20개 표본</h1><p>판단: Codex 직접 장면 검토 / 스크립트: 장면 추출·수기 결과 저장·배점 합산만 수행.</p><p><strong>전문가 확정 정답이나 모델 예측 결과가 아닙니다. 학습 승인은 0개입니다.</strong> 전체 영상을 연속 재생해 검토한 결과가 아니라 시간 표시 장면의 자세 전환을 검토했습니다. 푸시업 점수는 표시된 측면 구간에만 적용합니다.</p><p>17개에 점수 초안을 붙였고 3개는 관측·기준 불확실성으로 총점을 보류했습니다. 100점도 제한된 항목의 충족을 뜻하며 완벽한 자세를 뜻하지 않습니다.</p><p>참고 기준: <a href="https://www.nasm.org/resource-center/exercise-library/push-up">NASM 푸시업</a> · <a href="https://www.acefitness.org/resources/everyone/exercise-library/135/bodyweight-squat/">ACE 스쿼트</a>. 항목별 40·30·20·10 등의 배점은 이 기관이 제공한 점수가 아닌 프로젝트 초안입니다.</p>'
    (OUT/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>자세 점수 표본 라벨</title><style>body{font-family:system-ui;max-width:1200px;margin:30px auto;padding:0 20px;line-height:1.6;color:#172b4d}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;text-align:left}article{border-top:2px solid #ddd;margin-top:32px}img{width:100%}.score{font-weight:bold;font-size:1.3em}summary{cursor:pointer;padding:12px;background:#eaf4f7}</style>'+intro+''.join(cards)+'</html>')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':build()
