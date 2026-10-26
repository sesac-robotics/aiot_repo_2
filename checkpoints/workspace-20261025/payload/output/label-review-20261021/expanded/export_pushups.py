"""직접 검토한 푸시업 점수 초안을 QEVD 확장 형식으로 내보낸다."""
import collections
import hashlib
import json
from pathlib import Path

OUT=Path('/home/jayden/SeSACPAI-Mid/output/label-review-20261021/expanded')


def export():
    """중복·충돌·판단 불가 표본을 제외하고 실험용 라벨을 저장한다.

    Returns:
        None: pushups_labels.json과 제외 기록을 저장한다.
    Raises:
        AssertionError: 참여자·파일 중복이 분할을 넘거나 점수 형식이 잘못된 경우.
    """
    decisions=json.loads((OUT/'direct_assessments.json').read_text());labels=[];excluded=[]
    for n in range(1,49):
        row=json.loads((OUT/f'{n:03d}.json').read_text());d=decisions[str(n)]
        if d.get('excluded'):
            excluded.append(dict(id=row['id'],reason=d['reason']));continue
        levels=d['levels'];score=None if None in levels else sum(w*v/2 for w,v in zip([40,30,20,10],levels))
        image=OUT/f'{n:03d}.jpg';times=d['times']
        labels.append(dict(video_path=row['path'],labels=row['source_labels'],labels_descriptive=[d['note']],split=row['split'],
          schema_version='qevd-extended-posture-score-v1',id=row['id'],exercise='pushups',person=row['person'],
          segment=dict(start_frame=round(times[0]*row['evidence']['fps']),end_frame=round(times[-1]*row['evidence']['fps']),fps=row['evidence']['fps'],evidence_times_seconds=times),
          rubric_version='project-core-form-v1',criterion_levels=levels,criterion_weights=[40,30,20,10],
          criterion_ids=['trunk_line','descent_range','return_extension','support'],score=score,unit='project_core_form_0_to_100',
          reviewer='Codex',label_source='codex_direct_sampled_frame_review',source_aware_review=True,
          review_status='AI_reviewed_experimental_only',expert_validated=False,eligible_for_experimental_training=score is not None,
          evidence_image=str(image),image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),video_sha256=row['video_sha256'],
          temporal_sampling='36장 시간 표시 장면 중 선택한 상단–하단–상단 구간. 연속 재생 검토 아님.',
          score_scope='측면 핵심 자세 네 항목. 머리 정렬·좌우 비대칭·팔 벌림·속도는 미포함.',note_ko=d['note']))
    good=[r for r in labels if r['eligible_for_experimental_training']]
    assert len({r['video_sha256'] for r in good})==len(good)
    for key in ['person','video_sha256']:
        groups=collections.defaultdict(set)
        for r in good:groups[r[key]].add(r['split'])
        assert all(len(g)==1 for g in groups.values())
    (OUT/'pushups_labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2))
    (OUT/'pushups_exclusions.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2))
    print('실험용 직접 라벨',len(good),'분할',dict(collections.Counter(r['split'] for r in good)))


if __name__=='__main__':export()
