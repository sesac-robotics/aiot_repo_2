"""동기화된 정면·측면 검토 결과를 QEVD 확장 점수 형식으로 저장한다."""
import collections
import hashlib
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent


def export():
    """점수 초안과 보류 사유를 내보낸다.

    Returns:
        None: JSON 파일을 저장한다. 점수 단위는 프로젝트 기준 0~100점이다.
    Raises:
        AssertionError: 분할 누출·잘못된 구간·점수 형식이 발견된 경우.
    """
    decisions=json.loads((OUT/'squats_direct_assessments.json').read_text());labels=[];excluded=[];hashes={}
    for n in range(49,89):
        row=json.loads((OUT/f'{n:03d}.json').read_text());d=decisions[str(n)]
        if d.get('excluded'):
            excluded.append(dict(id=row['id'],**d));continue
        path=row['path']
        if path not in hashes:hashes[path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        levels=d['levels'];assert all(v in [0,1,2,None] for v in levels)
        score=None if None in levels else sum(w*v/2 for w,v in zip([30,20,30,20],levels))
        assert row['start_frame']<row['end_frame']
        labels.append(dict(video_path=path,labels=row['source_labels'],labels_descriptive=[d['note']],split=row['split'],schema_version='qevd-extended-posture-score-v1',id=row['id'],exercise='squats',person=row['person'],segment=dict(start_frame=row['start_frame'],end_frame=row['end_frame'],fps=row['evidence']['fps'],evidence_times_seconds=d['times']),rubric_version='project-core-form-v1',criterion_levels=levels,criterion_weights=[30,20,30,20],criterion_ids=['descent_range','return_extension','knee_foot','trunk_balance'],score=score,unit='project_core_form_0_to_100',reviewer='Codex',label_source='codex_direct_sampled_frame_review',source_aware_review=True,review_status='AI_reviewed_experimental_only',expert_validated=False,eligible_for_experimental_training=score is not None,evidence_images=[str(OUT/f'{n:03d}.jpg'),str(OUT/'squat_front'/f'{n:03d}.jpg')],video_sha256=hashes[path],temporal_sampling='동기화된 반복 구간의 측면 36장·정면 12장. 연속 재생 검토 아님.',score_scope='다중 시점 핵심 네 항목. 측면 단독 입력에서는 정면 오류가 관측되지 않을 수 있음.',note_ko=d['note']))
    for key in ['person','video_sha256']:
        groups=collections.defaultdict(set)
        for r in labels:groups[r[key]].add(r['split'])
        assert all(len(g)==1 for g in groups.values())
    assert len({(r['video_sha256'],r['segment']['start_frame'],r['segment']['end_frame']) for r in labels})==len(labels)
    (OUT/'squats_labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2))
    (OUT/'squats_exclusions.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2))
    print('점수 초안',sum(r['score'] is not None for r in labels),'보류',len(excluded)+sum(r['score'] is None for r in labels))


if __name__=='__main__':export()
