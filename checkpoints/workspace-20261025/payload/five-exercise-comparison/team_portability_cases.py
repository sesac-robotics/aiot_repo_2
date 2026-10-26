"""실제 보유 관절의 이식 전후 동일성 검사 입력을 Git 밖에 준비한다."""

import json
import pathlib

import cv2
import joblib
import numpy as np

import audit
import ytw_projected_rule_trial
from allfive_round1_train import manifest, predict
from held_round2 import OUT
from held_round2_train import added_rows

DEST = OUT.parent / "team_portability"


def prepare() -> dict:
    """직접 관측·추적 승인을 보존한 실제 구간을 수치 NPZ로 복사한다.

    Returns:
        dict: 학습에 사용한 구간의 특징·추론 동일성 검사 목록. 독립 시험이나
        새로운 직접 검토·정확도 측정으로 해석하지 않는다.

    Raises:
        ValueError: 직접 주석·원본·추적 승인 또는 캐시 계약 위반.
        OSError: 기존 입력·검사 출력 접근 실패.
    """
    DEST.mkdir(exist_ok=True)
    rows = {r["id"]: r for r in added_rows()}
    sources = {r['id']:r for r in audit.read_json(OUT/'dense_preparation.json')['records']}
    sources.update({r['id']:r for r in audit.read_json(OUT/'held_plank_preparation.json')['records']})
    receipts = {r['id']:r for r in audit.read_json(OUT/'extraction.json')['records']}
    reviews = {r['id']:r for r in audit.read_json(OUT/'subject_tracking_review.json')['records']}
    records=[]
    bundle=audit.BASE/'team-posture-scoring/Server/AIService/artifacts/posture_score'
    models={e:joblib.load(bundle/f'{e}.joblib') for e in ('pushups','squats','shoulder','plank','ytw')}
    for decision in audit.read_json(OUT/'manual_decisions.json')['records']:
        identifier=decision['id'];row=rows[identifier+'_held2_score'];e=row['exercise']
        if e=='plank' and not any('side' in v for v in decision['views']):
            continue
        rec=receipts[identifier]['extraction'];source=sources[identifier]
        with np.load(rec['cache']) as data:
            seq=data['image_xy'].copy();frames=data['frame_numbers'].copy()
        for a,b in reviews[identifier].get('unreliable_frame_intervals',[]):
            seq[(frames>=a)&(frames<=b)]=np.nan
        cap=cv2.VideoCapture(source['path']);width,height=int(cap.get(3)),int(cap.get(4));cap.release()
        target=DEST/(identifier+'.npz')
        np.savez_compressed(target,image_xy=seq,frame_numbers=frames,
            fps=source['fps'],width=width,height=height)
        expected=(ytw_projected_rule_trial.predict(models[e],manifest([row])[0]) if e=='ytw' else predict(models[e],row))
        permitted=[g is not None and i not in reviews[identifier].get('blocked_item_indices',[]) for i,g in enumerate(row['levels'])]
        options=dict(view=decision['views'][0],complete_attempt=decision['attempt_complete'],permitted_items=permitted)
        if options['view']=='side_oblique':
            options['view']='side'
        if e=='ytw':
            options.update(phase_windows=decision['phase_windows'],projection_reviewed=True)
        records.append(dict(id=identifier,exercise=e,landmarks=str(target),options=options,
            expected={k:expected[k] for k in ('ai_levels','rule_levels','ai_score','rule_score')},
            video_path=source['path'],fps=source['fps']))
    audit.save_json(DEST/'cases.json',dict(records=records,training_cases_not_accuracy_test=True))
    return dict(cases=len(records))


if __name__=='__main__':
    print(json.dumps(prepare()))
