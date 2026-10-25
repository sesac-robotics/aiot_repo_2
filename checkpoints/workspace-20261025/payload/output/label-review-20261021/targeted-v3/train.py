"""직접 보충한 항목 라벨로 고정 회귀 모델을 학습하고 기존 검증 분할에 비교한다."""
import collections
import json
from pathlib import Path
import sys
import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'expanded'
sys.path.insert(0,str(OUT.parent/'compact-v2'))
from experiment import compact,model
POLICY=dict(min_grade_samples=3,min_validation_samples=5,min_validation_grades=2,max_validation_MAE_grade=.5,min_validation_samples_per_grade=3,max_grade_specific_MAE=.5,min_relative_baseline_improvement=.1,require_group_cv_improvement=True)


def load(exercise,split):
    """라벨과 관절 특징을 연결한다.

    Args:
        exercise: pushups 또는 squats.
        split: train 또는 validation. 시험 분할은 허용하지 않는다.
    Returns:
        list: 원본 라벨·98차원 특징·출처를 포함한 표본 목록.
    Raises:
        AssertionError: 시험 접근 또는 중복 원본이 발견된 경우.
    """
    assert split in ['train','validation'];rows=[];hashes=set()
    sources=[(OLD/f'{exercise}_labels.json',OLD/'features'),(OUT/'labels.json',OUT/'features')]
    for source,folder in sources:
        for r in json.loads(source.read_text()):
            if r['exercise']!=exercise or r['split']!=split:continue
            path=folder/(r['id']+'.npz')
            if not path.exists():continue
            with np.load(path) as z:
                if not z['X'].size:continue
                x,names=compact(z['X'])
            key=(r['video_sha256'],r['segment']['start_frame'],r['segment']['end_frame']);assert key not in hashes;hashes.add(key)
            rows.append(dict(label=r,x=x,feature_path=str(path)))
    return rows


def train():
    """사람별 교차검증·고정 검증으로 항목의 웹 실험 허용 여부를 기록한다.

    Returns:
        None: 모델 파일, 지표 JSON과 사전 고정한 gate_policy.json을 저장한다.
    Raises:
        AssertionError: 학습·검증 사람 또는 원본 파일이 겹치는 경우.
    """
    (OUT/'gate_policy.json').write_text(json.dumps(POLICY,ensure_ascii=False,indent=2));report={};bundles={}
    for ex,allowed in [('pushups',['trunk_line','descent_range','return_extension','support']),('squats',['descent_range','return_extension'])]:
        tr=load(ex,'train');va=load(ex,'validation')
        for key in ['person','video_sha256']:
            assert not ({r['label'][key] for r in tr}&{r['label'][key] for r in va})
        results={};models={}
        for item in allowed:
            idx=tr[0]['label']['criterion_ids'].index(item);a=[r for r in tr if r['label']['criterion_levels'][idx] is not None];b=[r for r in va if r['label']['criterion_levels'][idx] is not None]
            x=np.array([r['x'] for r in a]);y=np.array([r['label']['criterion_levels'][idx] for r in a]);groups=np.array([r['label']['person'] for r in a]);counts=collections.Counter(int(v) for v in y)
            if len(counts)<2 or min(counts.values())<POLICY['min_grade_samples']:
                results[item]=dict(status='grade_diversity_insufficient',train_grades=dict(counts),web_experiment_enabled=False);continue
            oof=np.empty(len(y));base=np.empty(len(y));folds=[]
            for ti,vi in GroupKFold(3).split(x,y,groups):
                reg=model(37);weights=np.array([len(ti)/(len(set(y[ti]))*sum(y[ti]==v)) for v in y[ti]])
                reg.fit(x[ti],y[ti],extratreesregressor__sample_weight=weights);oof[vi]=reg.predict(x[vi]);base[vi]=y[ti].mean();folds.append(dict(validation_people=sorted(set(groups[vi])),train_grades=sorted(set(int(v) for v in y[ti]))))
            reg=model(37);weights=np.array([len(y)/(len(counts)*counts[int(v)]) for v in y]);reg.fit(x,y,extratreesregressor__sample_weight=weights);models[item]=reg;group_mae=float(mean_absolute_error(y,oof));group_base=float(mean_absolute_error(y,base));vx=np.array([r['x'] for r in b]);vy=np.array([r['label']['criterion_levels'][idx] for r in b]);pred=reg.predict(vx);vmae=float(mean_absolute_error(vy,pred));vbase=float(mean_absolute_error(vy,np.full(len(vy),y.mean())))
            per_grade={int(grade):dict(samples=int(sum(vy==grade)),MAE=float(mean_absolute_error(vy[vy==grade],pred[vy==grade]))) for grade in sorted(set(vy))}
            reasons=[]
            if any(v['samples']<POLICY['min_validation_samples_per_grade'] for v in per_grade.values()):reasons.append('등급별 검증 표본 부족')
            if any(v['MAE']>POLICY['max_grade_specific_MAE'] for v in per_grade.values()):reasons.append('오류 등급별 검증 오차 기준 미달')
            if len(b)<POLICY['min_validation_samples']:reasons.append('검증 표본 부족')
            if len(set(vy))<POLICY['min_validation_grades']:reasons.append('검증 등급 다양성 부족')
            if vmae>POLICY['max_validation_MAE_grade']:reasons.append('검증 오차 기준 미달')
            if vmae>vbase*(1-POLICY['min_relative_baseline_improvement']):reasons.append('평균값 기준 대비 검증 개선 부족')
            if group_mae>=group_base:reasons.append('참여자 교차검증 개선 부족')
            results[item]=dict(status='experimental_candidate' if not reasons else 'validation_failed',train_grades=dict(counts),train_samples=len(a),validation_samples=len(b),validation_grades=dict(collections.Counter(int(v) for v in vy)),group_MAE_grade=group_mae,group_baseline_MAE_grade=group_base,validation_MAE_grade=vmae,validation_baseline_MAE_grade=vbase,validation_per_grade=per_grade,web_experiment_enabled=not reasons,hold_reasons=reasons,folds=folds,validation_predictions=[dict(id=r['label']['id'],label=int(t),prediction=float(p)) for r,t,p in zip(b,vy,pred)])
        report[ex]=dict(train_features=len(tr),validation_features=len(va),train_people=len(set(r['label']['person'] for r in tr)),items=results)
        bundles[ex]=dict(models=models,metadata=results,feature_schema='compact_kinematics_98_v2',rubric_version='project-core-form-v1',expert_validated=False,full_score_enabled=False)
    joblib.dump(bundles,OUT/'item_models.joblib')
    data=dict(version='targeted-item-model-v4-balanced',test_split_used=False,expert_validated=False,production_ready=False,full_score_enabled=False,training_weighting='등급별 역빈도 가중치. 테스트 분할은 사용하지 않음.',gate_policy=POLICY,results=report,limitations=['기존 검증 분할을 여러 개발 실험에 사용했으므로 독립 최종 시험은 아님','추가 QEVD 영상은 정면 장면이 많아 측면 몸통 라벨을 부여하지 않음','0~2 항목 등급에 대한 모델 예측이며 종합 100점 평가가 아님'])
    (OUT/'training_report.json').write_text(json.dumps(data,ensure_ascii=False,indent=2));print(json.dumps(data,ensure_ascii=False,indent=2))


if __name__=='__main__':train()
