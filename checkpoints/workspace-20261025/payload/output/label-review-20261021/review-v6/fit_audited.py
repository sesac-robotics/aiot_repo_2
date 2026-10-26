"""재검토한 점수 초안으로 총점을 직접 예측하는 소규모 회귀 실험을 수행한다.

푸시업은 측면, 스쿼트는 동기화 정면·측면 특징을 사용한다. 점수는 관절각
규칙으로 생성하지 않으며, 실험 모델은 운영 웹의 활성 모델로 교체하지 않는다.
"""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from collections import Counter
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[2]
OLD=ROOT.parent/'expanded'
TARGET=ROOT.parent/'targeted-v3'
sys.path.insert(0,str(TARGET))
from features_2d import extract_one
from runtime_guard import temperature_check
from predict import compact
sys.path.insert(0,str(BASE/'overnight-posture-pipeline'))
from features import dense_clip


def features_for(row):
    """실제 구간·필요 시점에 맞는 특징을 가져오거나 새로 추출한다.

    Args:
        row: 재검토 라벨. 구간 단위는 원본 프레임 번호이다.
    Returns:
        ndarray 또는 None: 측면 210차원 또는 정면 포함 322차원 특징.
    Raises:
        RuntimeError: 고온·센서 실패 또는 캐시 불일치 시.
    """
    folder=ROOT/'features';folder.mkdir(exist_ok=True);target=folder/(row['id']+'.npz')
    key=dict(hash=row['video_sha256'],first=row['segment']['start_frame'],last=row['segment']['end_frame'],views=row['required_input_views'],front=row.get('front_path'))
    fingerprint=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
    if target.exists():
        with np.load(target) as z:
            assert str(z['fingerprint'])==fingerprint
            return z['X'] if z['X'].size else None
    temperature_check();world=None;image=None
    if not row['training_feature_cache_invalidated']:
        old_world=OLD/'features'/(row['id']+'.npz');old_image=TARGET/'image_features'/(row['id']+'.npz')
        if old_world.exists():
            with np.load(old_world) as z:world=z['X'].copy()
            if not world.size:return None
        if old_image.exists():
            with np.load(old_image) as z:image=z['X'].copy()
    if world is None:
        source=dict(path=row['video_path'],start_frame=row['segment']['start_frame'],end_frame=row['segment']['end_frame'])
        world,_,_=dense_clip(source)
        if world is None:
            np.savez_compressed(target,X=[],fingerprint=fingerprint);return None
    if image is None:image,_,_,_=extract_one(row)
    values=[compact(world)[0],image]
    if row['exercise']=='squats':
        front=copy.deepcopy(row);front['video_path']=row['front_path'];x,_,_,coverage=extract_one(front)
        if coverage<.35:
            np.savez_compressed(target,X=[],fingerprint=fingerprint);return None
        values.append(x)
    result=np.concatenate(values);np.savez_compressed(target,X=result,fingerprint=fingerprint);return result


def model(seed=37):
    """결측 처리를 포함한 고정 설정의 총점 회귀 모델을 반환한다.

    Args:
        seed: 난수 시드. 물리 단위 없음.
    Returns:
        Pipeline: 중앙값 결측 보완과 ExtraTrees 회귀 파이프라인.
    """
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),ExtraTreesRegressor(n_estimators=250,min_samples_leaf=2,max_features=.7,n_jobs=2,random_state=seed))


def main():
    """학습 사람별 검증과 기존 검증 분할 평가를 저장한다.

    Returns:
        None: 모델·오차·표본별 예측·특징 처리 결과를 저장한다.
    Raises:
        AssertionError: 사람 또는 영상의 학습·검증 중복 시.
    """
    extraction=[];report={};models={}
    for ex in ['pushups','squats']:
        labels=json.loads((ROOT/f'{ex}_labels.json').read_text());rows=[]
        for r in labels:
            if r['split']=='test' or not r['eligible_for_total_score_training']:continue
            x=features_for(r);extraction.append(dict(id=r['id'],split=r['split'],valid=x is not None,cache_invalidated=r['training_feature_cache_invalidated']))
            (ROOT/'extraction_report.json').write_text(json.dumps(extraction,ensure_ascii=False,indent=2))
            print(ex,r['id'],x is not None,flush=True)
            if x is not None:rows.append(dict(label=r,x=x))
        tr=[r for r in rows if r['label']['split']=='train'];va=[r for r in rows if r['label']['split']=='validation']
        for key in ['person','video_sha256']:assert not ({r['label'][key] for r in tr}&{r['label'][key] for r in va})
        x=np.array([r['x'] for r in tr]);y=np.array([r['label']['score'] for r in tr]);g=np.array([r['label']['person'] for r in tr]);vx=np.array([r['x'] for r in va]);vy=np.array([r['label']['score'] for r in va]);oof=np.empty(len(y));baseline=np.empty(len(y));curve=[]
        for ti,vi in GroupKFold(min(3,len(set(g)))).split(x,y,g):
            fitted=model();fitted.fit(x[ti],y[ti]);oof[vi]=fitted.predict(x[vi]);baseline[vi]=np.median(y[ti])
        for cap in [6,10,len(y)]:
            if cap>len(y):continue
            predictions=[]
            for seed in [11,23,37]:
                pred=np.empty(len(y))
                for ti,vi in GroupKFold(min(3,len(set(g)))).split(x,y,g):
                    selected=np.random.default_rng(seed).permutation(ti)[:min(cap,len(ti))];fit=model(seed);fit.fit(x[selected],y[selected]);pred[vi]=fit.predict(x[vi])
                predictions.append(float(mean_absolute_error(y,pred)))
            curve.append(dict(max_train_samples_per_fold=cap,MAE_points=predictions))
        fitted=model();fitted.fit(x,y);p=fitted.predict(vx);models[ex]=dict(model=fitted,required_input_views=tr[0]['label']['required_input_views'],feature_count=x.shape[1],target='Codex_rechecked_project_score_0_100',production_ready=False)
        report[ex]=dict(train_samples=len(tr),train_people=len(set(g)),validation_samples=len(va),validation_people=len(set(r['label']['person'] for r in va)),train_score_counts=dict(Counter(y)),participant_CV_MAE_points=float(mean_absolute_error(y,oof)),participant_CV_baseline_MAE_points=float(mean_absolute_error(y,baseline)),validation_MAE_points=float(mean_absolute_error(vy,p)),validation_baseline_MAE_points=float(mean_absolute_error(vy,np.full(len(vy),np.median(y)))),validation_max_absolute_error_points=float(np.max(np.abs(vy-p))),validation_predictions=[dict(id=r['label']['id'],label=float(t),prediction=float(v),absolute_error=float(abs(t-v))) for r,t,v in zip(va,vy,p)],learning_curve=curve)
    output=dict(version='audited-total-score-regression-v6',target_source='Codex 직접 재검토 프로젝트 점수 초안',prediction_method='회귀 모델이 총점 0~100을 직접 예측. 추론 시 규칙 합산·규칙 대체 없음.',test_features_or_predictions_used=False,validation_is_independent_final_test=False,production_ready=False,full_score_web_enabled=False,results=report,limitations=['기존 검증 사람을 반복 사용한 개발 실험이다.','푸시업 손잡이 도메인·구간 불명확·시각 근거 부족 사례는 총점 학습에서 제외했다.','스쿼트는 정면·측면 두 시점이 필요하며 측면 단독 웹에 이 총점 모델을 적용하지 않는다.','라벨은 직접 프레임 검토한 실험 기준이며 전문가 정답 정확도로 해석하지 않는다.','자료 수를 늘려도 개선이 없는 지점은 이 소규모 실험만으로 확정하지 않는다.'])
    joblib.dump(models,ROOT/'total_score_models.joblib');(ROOT/'training_report.json').write_text(json.dumps(output,ensure_ascii=False,indent=2));print(json.dumps(output,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
