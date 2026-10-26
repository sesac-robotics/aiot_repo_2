"""직접 검토한 항목별 라벨을 QEVD 확장 형식으로 저장한다."""
import hashlib
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent


def export():
    """관측 불가 항목을 보존한 채 라벨과 증거를 내보낸다.

    Returns:
        None: labels.json에 0~2 등급과 가능한 경우 0~100점 초안을 저장한다.
    Raises:
        AssertionError: 중복·구간·등급 형식이 잘못된 경우.
    """
    decisions=json.loads((OUT/'direct_assessments.json').read_text());labels=[]
    for row in json.loads((OUT/'manifest.json').read_text()):
        d=decisions[str(row['review_number'])];push=row['exercise']=='pushups';weights=[40,30,20,10] if push else [30,20,30,20]
        ids=['trunk_line','descent_range','return_extension','support'] if push else ['descent_range','return_extension','knee_foot','trunk_balance'];levels=d['levels'];times=d['times'];fps=row['evidence']['fps']
        assert all(v in [0,1,2,None] for v in levels)
        start=round(times[0]*fps);end=round(times[-1]*fps);assert 0<=start<end<row['evidence']['frame_count']
        labels.append(dict(schema_version='qevd-extended-posture-score-v1',id=row['id'],exercise=row['exercise'],person=row['person'],split=row['split'],video_path=row['path'],video_sha256=row['video_sha256'],labels=row['source_labels'],labels_descriptive=[d['note']],criterion_ids=ids,criterion_weights=weights,criterion_levels=levels,score=None if None in levels else sum(w*v/2 for w,v in zip(weights,levels)),unit='project_core_form_0_to_100',rubric_version='project-core-form-v1',eligible_for_experimental_training=any(v is not None for v in levels),eligible_for_total_score_training=None not in levels,label_source='codex_direct_sampled_frame_review',source_aware_review=True,expert_validated=False,review_status='AI_reviewed_experimental_only',view=d['view'],evidence_image=row['evidence']['image'],image_sha256=hashlib.sha256(Path(row['evidence']['image']).read_bytes()).hexdigest(),segment=dict(start_frame=start,end_frame=end,fps=fps,evidence_times_seconds=times),note_ko=d['note']))
    assert len({r['video_sha256'] for r in labels})==len(labels)
    (OUT/'labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2));print('항목별 직접 라벨',len(labels),'총점 가능',sum(r['score'] is not None for r in labels))


if __name__=='__main__':export()
