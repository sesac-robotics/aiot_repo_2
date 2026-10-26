"""직접 점수 초안·관절 검토·관측 교집합을 학습용 QEVD 확장 행으로 만든다."""
import collections
import hashlib
import json
import pathlib

from app.scoring.engine import ScoringEngine
from app.scoring.features import DEFINITIONS, describe
from app.scoring.media import load_landmarks

ROOT=pathlib.Path('/home/jayden/SeSACPAI-Mid/output/five-exercise-recovery-20261023')


def main() -> None:
    """직접 검토하지 않은 관절이나 누락 항목을 학습 예제로 채우지 않는다."""
    source={r['id']:r for r in json.loads((ROOT/'screening_manifest.json').read_text())['records']}
    decisions=json.loads((ROOT/'manual_decisions.json').read_text())['records']
    reviews={r['id']:r for r in json.loads((ROOT/'subject_tracking_review.json').read_text())['records']}
    if set(reviews)!={r['id'] for r in decisions}:raise ValueError('Every draft needs explicit tracking review before training')
    scorer=ScoringEngine();rows=[];annotations=[]
    for d in decisions:
        r=source[d['id']];review=reviews[d['id']]
        if not review['direct_temporal_tracking_review']:raise ValueError('Missing direct tracking review')
        if hashlib.sha256(pathlib.Path(r['path']).read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Source changed after review')
        seq=load_landmarks(ROOT/'observations'/d['id']/'primary.npz')
        if int(seq.frame_numbers[0])!=d['frames'][0] or seq.frame_numbers[-1]>d['frames'][1]:raise ValueError('Review and extraction interval differ')
        notes=describe(r['exercise'],seq,scorer.model(r['exercise']),view=d['view'],phase_windows=d['phase_windows'],projection_reviewed=d['projection_reviewed'])
        for i,n in enumerate(notes):
            if i in review['blocked_item_indices'] or d['levels'][i] is None:
                n['eligible']=False;n['holds'].append('direct_review_item_unobservable_or_tracking_unreliable')
            if not n['eligible']:n['feature']=None
        groups=[r.get('recording_group',r['person'])]
        if d['id'] in ('qevd_00258184','qevd_00079260'):groups.append('possible_same_yellow_room_performer_round3')
        row=dict(id=d['id']+'_held3_score',exercise=r['exercise'],person=r['person'],split='train',sources=list(dict.fromkeys([r['sha256']]+([r['original_source_sha256']] if 'original_source_sha256' in r else []))),recording_groups=groups,levels=d['levels'],reference_total_allowed=d['attempt_complete'] and None not in d['levels'],whole_attempt_observed=d['attempt_complete'],items=notes,label_origin='direct_visual_draft')
        rows.append(row);ids,weights=DEFINITIONS[r['exercise']]
        annotations.append(dict(video_path=r['path'],labels=r['source_labels'],labels_descriptive=[d['note_ko']],split='train',schema_version='qevd-extended-posture-score-v2',id=row['id'],exercise=r['exercise'],person=r['person'],segment=dict(start_frame=d['frames'][0],end_frame=d['frames'][1],fps=r['fps']),criterion_ids=ids,criterion_weights=weights,criterion_levels=d['levels'],score=sum(w*g/2 for w,g in zip(weights,d['levels'])) if row['reference_total_allowed'] else None,score_is_expert_ground_truth=False,label_source='AI_direct_chronological_sampled_frame_draft',rule_generated_label=False,tracking_approved_item_indices=[i for i in range(4) if i not in review['blocked_item_indices']],phase_windows=d['phase_windows'],source_sha256=r['sha256'],review_evidence=review['evidence'],raw_review_evidence=str(ROOT/'dense'/(d['id']+'.jpg'))))
    (ROOT/'new_feature_rows.json').write_text(json.dumps(dict(records=rows),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (ROOT/'labels.json').write_text(json.dumps(annotations,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    effective=[r for r in rows if any(n['eligible'] and g is not None for n,g in zip(r['items'],r['levels']))]
    full=[r for r in effective if r['reference_total_allowed'] and all(n['eligible'] for n in r['items'])]
    summary=dict(direct_draft_rows=len(rows),tracking_reviewed_rows=len(reviews),effective_new_training_rows=dict(collections.Counter(r['exercise'] for r in effective)),full_reference_and_input_rows=dict(collections.Counter(r['exercise'] for r in full)),plank_new_core_labels=sum(r['exercise']=='plank' for r in effective),plank_complete_3s_windows=0,plank_support_contrast_rows=1,plank_short_hold_rows=1,screened_original_candidates=43,independent_final_test=False,external_discovery=False)
    (ROOT/'label_qualification.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)


if __name__=='__main__':main()
