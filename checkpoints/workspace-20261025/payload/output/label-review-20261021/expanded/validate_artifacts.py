"""라벨·특징·저장 모델의 연결과 누출 방지를 검증한다."""
import collections
import csv
import json
from pathlib import Path
import joblib
import numpy as np
OUT=Path(__file__).resolve().parent


def validate():
    """산출물을 검사하고 팀 검토용 CSV와 배포 보류 사유를 저장한다.

    Returns:
        None: verification.json, review_queue.csv, release_gate.json을 저장한다.
    Raises:
        AssertionError: 점수 누락·분할 누출·비정상 모델 출력이 발견된 경우.
    """
    results={};queue=[]
    for ex in ['pushups','squats']:
        rows=json.loads((OUT/f'{ex}_labels.json').read_text());artifact=joblib.load(OUT/'models'/f'{ex}_direct_draft.joblib');valid=[];groups=collections.defaultdict(set)
        for r in rows:
            groups[r['person']].add(r['split'])
            assert r['expert_validated'] is False
            assert (r['score'] is None)==(None in r['criterion_levels'])
            assert r['eligible_for_experimental_training']==(r['score'] is not None)
            if r['score'] is None:continue
            assert r['score']==sum(w*v/2 for w,v in zip(r['criterion_weights'],r['criterion_levels']))
            assert r['segment']['end_frame']>r['segment']['start_frame']>=0
            p=OUT/'features'/(r['id']+'.npz')
            if p.exists():
                with np.load(p) as z:
                    if z['X'].size:
                        prediction=float(artifact['model'].predict(z['X'][None,:])[0]);assert np.isfinite(prediction) and 0<=prediction<=100;valid.append(r['id'])
            queue.append(dict(exercise=ex,id=r['id'],split=r['split'],AI_draft_score=r['score'],expert_score='',reviewer='',review_notes='',evidence=r.get('evidence_image') or '|'.join(r['evidence_images'])))
        assert all(len(v)==1 for v in groups.values())
        results[ex]=dict(labels_with_score=sum(r['score'] is not None for r in rows),valid_features_and_predictions=len(valid),subject_split_disjoint=True,expert_accuracy=None)
    with (OUT/'review_queue.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(queue[0]));writer.writeheader();writer.writerows(queue)
    gate=dict(ready_for_production=False,large_scale_label_expansion='hold_until_calibration',reasons=['두 종목의 전체 학습 표본 검증 MAE가 평균값 비교 기준보다 큼','독립 전문가 라벨 검증 없음','스쿼트 다중 시점 라벨과 측면 단독 입력의 관측 범위 차이','현재 시험 집합은 이미 조회되어 새 독립 검증 필요'],next_actions=['항목별 관측 가능성과 배점 검토','참여자 단위 교차검증 및 불확실한 라벨 재검토','팀 대표 표본 검토 후 종목별 라벨 확대'],remaining_exercises=['plank','ytw','shoulder'],old_labels_reused=0)
    (OUT/'release_gate.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2))
    (OUT/'verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(json.dumps(results,ensure_ascii=False))


if __name__=='__main__':validate()
