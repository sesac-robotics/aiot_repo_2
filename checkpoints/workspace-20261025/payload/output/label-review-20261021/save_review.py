"""Codex가 직접 확인한 표본 장면의 관찰 근거와 QEVD 호환 검토 자료를 저장한다."""
from pathlib import Path
import json
import html

OUT=Path('/home/jayden/SeSACPAI-Mid/output/label-review-20261021')
notes={
1:('0.00–19.80초에 팔 굽힘·펴기와 몸통 높이 변화가 보인다. 마지막 21.80초는 무릎을 내려 종료하는 구간이다.','반복 경계와 종료 구간을 분리해야 한다.'),
2:('1.93초 팔을 편 장면과 9.93초 깊게 내려간 장면이 구분된다. 종료 21.93초는 운동 수행에서 제외해야 한다.','측면만으로 팔 벌림·좌우 비대칭을 확정할 수 없다.'),
3:('3.00–13.47초에 내려감·올라감 상태가 구분되나 세로 영상 안의 인물이 작다.','원본에서 인물 확대 및 촘촘한 시간 검토가 필요하다.'),
4:('0.00–3.47초 내려간 자세와 5.20–10.47초 팔을 편 자세가 보인다. 19.27초에는 무릎을 내린다.','균일 샘플링만으로 반복 깊이의 최솟값을 보장하지 못한다.'),
5:('0.00초 팔을 편 상태, 5.20–15초대 내려간 상태가 보인다. 얼굴 마스킹과 주변 가림이 있다.','머리 정렬 및 후반부 일부 자세는 판단 보류한다.'),
6:('0.00–16.40초의 여러 장면에서 엉덩이가 어깨–발목 연결선보다 높게 올라간 모습이 반복된다.','엉덩이 들림의 시각적 근거가 있다. 연속 영상으로 지속 구간을 확인해야 한다.'),
7:('1.47–3.00초와 7.60–10.67초에 팔 굽힘 상태가 보인다. 몸통·발은 화면에 있다.','bad 분류의 구체적 원인은 이 측면 샘플만으로 확정하지 않는다.'),
8:('여러 시점에서 굽힌 팔과 편 팔이 구분되나 인물 크기가 작다.','bad 분류의 원인을 추정하지 않는다. 확대 및 다른 시점이 필요하다.'),
9:('1.20–3.73초와 6.20–8.73초에 상체 높이 변화가 보인다. 일부 장면에서 목·몸통 정렬 차이가 의심된다.','오류 종류·정도는 확정 보류한다. 제공자 bad 분류와 관찰을 분리한다.'),
10:('측면 장면에서 팔 굽힘·펴기는 보이나 얼굴·머리가 마스킹 또는 화면 가장자리와 겹친다.','머리 정렬과 반복 최저점을 확정하지 않는다.'),
11:('0.00초 서기, 1.07–1.87초 낮은 자세, 2.67–2.97초 서기 순서가 보인다.','REHAB26 정상 라벨은 재활 과제 기준이다. 일반 근력 스쿼트의 95점과 동일시하지 않는다.'),
12:('0.00초 서기에서 0.53–0.80초 하강 후 1.50초 다시 선다.','재활 기준 정상 여부와 프로젝트 깊이 배점은 별도 검토해야 한다.'),
13:('0.00초부터 하강하여 0.73–1.10초 낮은 자세, 1.87–2.07초 서기로 이어진다.','무릎 좌우 정렬은 이 시점 하나로 충분히 평가하기 어렵다.'),
14:('0.00초 서기, 0.80–1.23초 낮은 자세, 2.27초 서기로 이어진다.','발 접지 및 무릎 정렬은 원본 확대나 다른 카메라와 대조해야 한다.'),
15:('0.00초 서기, 1.27–1.93초 낮은 자세, 3.57초 서기가 보인다.','서로 다른 체형·관점으로 절대 깊이 기준을 단정하지 않는다.'),
16:('0.70–1.07초 낮은 자세에서 무릎의 모임 가능성이 보이나 발·무릎이 일부 겹친다.','원본 ERR_KNEES_INSIDE는 보존한다. 시각 확정은 다른 카메라가 필요하다.'),
17:('0.97–1.43초 낮은 자세를 지나 다시 선다. 현재 시점에서는 좌우 기울기 크기를 분명히 구별하기 어렵다.','원본 ERR_TORSO_TO_SIDE를 보존하고 정면 카메라 대조 전 확정을 보류한다.'),
18:('0.97–1.47초 하강 구간에서 상체 방향 변화가 보인다.','원본 ERR_TORSO_TO_SIDE의 방향·크기는 정면 또는 다른 시점과 대조한다.'),
19:('0.00–1.40초에서 비교적 작은 하강 후 다시 선다. 같은 참여자의 13번보다 하강 범위가 작아 보인다.','원본 ERR_SHALLOW_SQUAT와 방향상 부합한다. 동일 촬영 방향 확인 후 정량 비교한다.'),
20:('0.53–0.97초 하강 구간에서 무릎이 발 앞쪽으로 이동한 모습이 보인다.','ERR_KNEES_FORWARD는 제공자 정의다. 무릎이 발끝을 지났다는 사실만으로 일반 운동 오류나 부상을 단정하지 않는다.')}


def save():
    """20개 장면 검토와 제안 채점표를 저장한다. 연속 점수 정답은 생성하지 않는다.

    Returns:
        None: JSON과 로컬 HTML 보고서를 기록한다.
    Raises:
        OSError: 파일 읽기·쓰기 실패.
    """
    manifest=json.loads((OUT/'manifest.json').read_text());reviews=[]
    for r in manifest:
        evidence,limit=notes[r['review_id']]
        reviews.append(dict(video_path=r['path'],labels=r['source_labels'],labels_descriptive=[evidence],split=r['split'],
            schema_version='qevd-compatible-review-v1',clip_id=r['id'],exercise=r['exercise'],participant_id=r['person'],
            source=dict(dataset=r['label_source'],provider_labels=r['source_labels'],provider_class=r['provider_class']),
            segment=dict(start_frame=r.get('start_frame'),end_frame=r.get('end_frame')),
            review=dict(reviewer='Codex',method='12_timestamped_still_frames_plus_provider_annotation_comparison',
                        blind_review=False,full_video_watched=False,evidence_ko=evidence,limitations_ko=limit,
                        image=r['montage_path'],status='preliminary_evidence_review_not_expert_ground_truth'),
            posture_score=dict(value=None,unit='0_to_100',status='withheld_pending_rubric_and_temporal_multiview_checks'),
            eligible_for_training=False))
    (OUT/'reviews.json').write_text(json.dumps(reviews,ensure_ascii=False,indent=2))
    rubric=dict(version='proposed-v1',status='unvalidated_project_rubric',score_is_not_provider_ground_truth=True,
        categories=dict(pushups=['몸통 정렬','팔 굽힘 범위','다시 팔 펴기','좌우 균형'],squats=['하강 범위','다시 일어서기','무릎·발 정렬','몸통 균형']),
        levels={'2':'해당 기준을 충족하는 시각 근거가 있음','1':'해당 기준의 부분적 미충족 근거가 있음','0':'해당 기준의 명확한 미충족 근거가 있음','null':'가림·관점·시간 표본 부족으로 판단 불가'},
        scoring='각 항목을 먼저 독립 평가. 모든 항목의 관측·동작 구간·정의가 확정된 경우에만 합계/최대합계×100. null 항목은 만점 처리하거나 나머지로 재정규화하지 않음.',
        timing='준비·종료 구간 제외. 반복별 시작·최저점·완료를 충분한 시간 해상도로 확인. 정지 장면으로 속도·제어를 추정하지 않음.',
        interpretation='배점은 프로젝트의 합의 전 제안이며 임상 또는 전문가 검증 점수 아님. 제공자의 정상·오류 이진값을 100·0점으로 바꾸지 않음.',
        validation='원본 오류 종류와 시각 근거의 일치·불일치·판단 불가를 기록. 원본을 본 검토이므로 독립 정확도 수치 산출 금지. 참여자·원본 반복 그룹별 분리 유지.')
    (OUT/'rubric_proposal.json').write_text(json.dumps(rubric,ensure_ascii=False,indent=2))
    cards=[]
    for r in reviews:
        v=r['review'];cards.append(f'<article><h2>{html.escape(r["clip_id"])}</h2><p>제공자: {html.escape(str(r["labels"]))}</p><img src="{Path(v["image"]).name}"><p>{html.escape(v["evidence_ko"])}</p><p>제약: {html.escape(v["limitations_ko"])}</p><p>점수 미확정 · 학습 제외</p></article>')
    (OUT/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>자세 라벨 표본 검토</title><style>body{font-family:sans-serif;max-width:1280px;margin:30px auto}img{width:100%}article{border-top:1px solid #aaa;padding:20px 0}</style><h1>푸시업·스쿼트 20개 표본의 초기 근거 검토</h1><p>Codex가 시간 표시 정지 장면을 직접 확인하고 원본 라벨과 대조했습니다. 전체 영상 재생 검토·전문가 확정·독립 정확도 시험을 완료한 결과는 아닙니다. 기존 자동 점수 119개는 학습 보류 상태입니다.</p>'+''.join(cards)+'</html>')


if __name__=='__main__':save()
