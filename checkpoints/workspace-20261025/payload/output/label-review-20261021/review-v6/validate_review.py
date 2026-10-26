"""재검토 라벨·재추출 캐시·모델 입력 분리의 일관성을 검증한다."""
import json
import hashlib
from pathlib import Path
import numpy as np
import joblib
ROOT=Path(__file__).resolve().parent


def validate():
    """관측 누락·구간 변경 캐시·학습 분할 오류 여부를 검사한다.

    Returns:
        dict: 검사 통과 여부와 건수. 점수의 실제 정확도를 검증하지는 않는다.
    Raises:
        AssertionError: 점수 합산·보류 처리·참여자 분리·캐시·예측 오류 발생 시.
    """
    rows=[]
    for ex in ['pushups','squats']:rows+=json.loads((ROOT/f'{ex}_labels.json').read_text())
    for r in rows:
        levels=r['criterion_levels']
        if None in levels:assert r['score'] is None and not r['eligible_for_total_score_training']
        else:assert r['score']==sum(w*v/2 for w,v in zip(r['criterion_weights'],levels))
        if r['total_score_hold_reason']:assert not r['eligible_for_total_score_training']
        if r['exercise']=='squats':assert r['required_input_views']==['side','synchronized_front']
        if r['training_feature_cache_invalidated']:assert [r['segment']['start_frame'],r['segment']['end_frame']]!=[r['original_segment']['start_frame'],r['original_segment']['end_frame']]
        assert all(Path(p).is_file() for p in r['audit_evidence'])
    for key in ['person','video_sha256']:
        for split in ['validation','test']:
            assert not ({r[key] for r in rows if r['split']=='train'} & {r[key] for r in rows if r['split']==split})
    extraction=json.loads((ROOT/'extraction_report.json').read_text());by_id={r['id']:r for r in rows}
    assert all(r['split']!='test' and by_id[r['id']]['eligible_for_total_score_training'] for r in extraction)
    models=joblib.load(ROOT/'total_score_models.joblib');report=json.loads((ROOT/'training_report.json').read_text())
    for ex,bundle in models.items():
        assert not bundle['production_ready']
        for sample in extraction:
            r=by_id[sample['id']]
            if r['exercise']!=ex or not sample['valid']:continue
            with np.load(ROOT/'features'/(r['id']+'.npz')) as z:
                x=z['X'];assert x.size==bundle['feature_count']
                key=dict(hash=r['video_sha256'],first=r['segment']['start_frame'],last=r['segment']['end_frame'],views=r['required_input_views'],front=r.get('front_path'))
                expected=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
                assert str(z['fingerprint'])==expected
            pred=bundle['model'].predict([x]);assert np.isfinite(pred).all() and 0<=pred[0]<=100
        for v in report['results'][ex]['validation_predictions']:assert np.isfinite(v['prediction']) and 0<=v['prediction']<=100
    result=dict(passed=True,reviewed=len(rows),extraction_attempts=len(extraction),test_extraction_count=0,missing_items_never_filled=True,changed_train_validation_segments_recomputed=True,required_views_checked=True,not_an_accuracy_validation=True)
    (ROOT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False));return result

if __name__=='__main__':validate()
