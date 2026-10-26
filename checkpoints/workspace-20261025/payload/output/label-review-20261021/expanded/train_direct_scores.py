"""직접 검토한 점수 초안으로 관절 입력 회귀 모델과 소규모 학습 곡선을 만든다.

시험 자료는 검증 단계의 모델·표본 수 선택을 마친 뒤 한 번만 평가한다.
결과는 Codex 프로젝트 라벨과의 일치도이며 전문가 채점 정확도가 아니다.
"""
from pathlib import Path
import collections
import json
import time
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error

OUT=Path('/home/jayden/SeSACPAI-Mid/output/label-review-20261021/expanded')


def model(seed):
    """시드가 고정된 작은 CPU 회귀 모델을 반환한다. 점수 규칙은 입력하지 않는다."""
    return make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),ExtraTreesRegressor(n_estimators=200,min_samples_leaf=2,max_features=.7,n_jobs=2,random_state=seed))


def train(exercise='pushups'):
    """관절 특징·점수 라벨을 연결해 참여자와 중복 파일 누출을 점검하고 학습한다.

    Args:
        exercise: 라벨 파일명에 쓰이는 영어 운동 코드.
    Returns:
        None: joblib 모델과 검증 곡선·시험 평가 JSON을 저장한다.
    Raises:
        AssertionError: 참여자·원본 파일 누출이 있거나 점수 라벨이 부족한 경우.
    """
    labels=json.loads((OUT/f'{exercise}_labels.json').read_text());rows=[];xs=[]
    for r in labels:
        p=OUT/'features'/(r['id']+'.npz')
        if r.get('eligible_for_experimental_training') and p.exists():
            with np.load(p) as z:
                if z['X'].size:rows.append(r);xs.append(z['X'])
    for key in ['person','video_sha256']:
        seen=collections.defaultdict(set)
        for r in rows:seen[r[key]].add(r['split'])
        assert all(len(v)==1 for v in seen.values()),'분할 누출'
    x=np.asarray(xs);y=np.array([r['score'] for r in rows],float);splits=np.array([r['split'] for r in rows]);tr=np.flatnonzero(splits=='train');va=np.flatnonzero(splits=='validation');te=np.flatnonzero(splits=='test')
    assert len(tr)>=15 and len(va)>=5 and len(te)>=5,'분할별 라벨 부족'
    assert len(set(y[tr]))>=3,'점수 분포 부족'
    sizes=sorted(set([min(8,len(tr)),min(14,len(tr)),len(tr)]));seeds=[37,71,109,151,193];curve=[]
    for size in sizes:
        errors=[];baselines=[]
        for seed in seeds:
            chosen=np.random.default_rng(seed).permutation(tr)[:size]
            reg=model(seed);reg.fit(x[chosen],y[chosen]);errors.append(float(mean_absolute_error(y[va],reg.predict(x[va]))));baselines.append(float(mean_absolute_error(y[va],np.full(len(va),y[chosen].mean()))))
        curve.append(dict(train_samples=size,validation_MAE_mean=float(np.mean(errors)),validation_MAE_std=float(np.std(errors)),validation_MAE_runs=errors,mean_baseline_MAE=float(np.mean(baselines))))
    best=min(curve,key=lambda c:c['validation_MAE_mean']);selected=next(c for c in curve if c['validation_MAE_mean']<=best['validation_MAE_mean']+2)
    chosen=np.random.default_rng(37).permutation(tr)[:selected['train_samples']]
    reg=model(37);start=time.monotonic();reg.fit(x[chosen],y[chosen]);seconds=time.monotonic()-start
    destination=OUT/'models'/f'{exercise}_direct_draft.joblib';destination.parent.mkdir(exist_ok=True)
    joblib.dump(dict(model=reg,exercise=exercise,rubric_version='project-core-form-v1',label_source='codex_direct_sampled_frame_review',expert_validated=False,unit='project_core_form_0_to_100',feature_schema='Heavy_10fps_BODY17_world_normalized_32timepoints_stats',feature_dimensions=x.shape[1],score_scope='라벨에 명시된 핵심 자세 항목에 대한 실험적 예측'),destination)
    # 시험은 표본 수를 선택한 이후 이 지점에서만 사용한다.
    predictions=reg.predict(x[te]);baseline=np.full(len(te),y[chosen].mean())
    improvements=[curve[i-1]['validation_MAE_mean']-curve[i]['validation_MAE_mean'] for i in range(1,len(curve))]
    report=dict(exercise=exercise,label_source='Codex_project_rubric_drafts',expert_accuracy=None,
        counts=dict(collections.Counter(r['split'] for r in rows)),feature_dimensions=x.shape[1],learning_curve=curve,
        selected_training_samples=len(chosen),selection_policy='검증 평균 MAE 최저값에서 2점 이내인 가장 작은 표본 수. 시험 자료는 선택에 사용하지 않음.',
        plateau=dict(improvements=improvements,status='inconclusive_small_validation_set',reason='검증 참여자·점수 범위가 작으므로 성능 정체를 확정하거나 전체 데이터가 불필요하다고 단정할 수 없음.'),
        test_MAE_points=float(mean_absolute_error(y[te],predictions)),test_RMSE_points=float(np.sqrt(mean_squared_error(y[te],predictions))),test_mean_baseline_MAE_points=float(mean_absolute_error(y[te],baseline)),
        test_predictions=[dict(id=rows[i]['id'],person=rows[i]['person'],draft_label=float(y[i]),model_prediction=float(p)) for i,p in zip(te,predictions)],
        model_path=str(destination),fit_seconds=seconds,
        limitations=['독립 전문가 점수 검증 없음','정지 장면의 시각적 판단과 제안 배점에 의존','사람별·촬영 환경별 표본 부족','실험 모델이며 기존 웹에 자동 배포하지 않음'])
    (OUT/f'{exercise}_training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    import sys
    train(sys.argv[1] if len(sys.argv)>1 else 'pushups')
