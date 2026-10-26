"""원본 화면의 기하 특징을 항목별로 선택해 고정된 사람별 검증으로 비교한다."""
import collections
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from train import load,model,POLICY
from predict import compact
OUT=Path(__file__).resolve().parent
CHANNELS={'trunk_line':['body_line_','hip_left_','hip_right_'],'support':['knee_left_','knee_right_','hip_ankle_distance_','knee_toe_height_'],'squat_depth':['knee_left_','knee_right_','hip_left_','hip_right_','hip_knee_height_','hip_ankle_height_']}


def fit(x,y):
    """등급별 역빈도로 가중한 회귀 모델을 반환한다.

    Args:
        x: 관절 특징 배열.
        y: 직접 검토한 0~2 항목 등급.
    Returns:
        Pipeline: 학습된 결측 처리·ExtraTrees 회귀 파이프라인.
    """
    counts=collections.Counter(y);weights=np.array([len(y)/(len(counts)*counts[v]) for v in y]);reg=model(37);reg.fit(x,y,extratreesregressor__sample_weight=weights);return reg


def expanded(ex,split):
    """동일 영상의 3D·2D 특징을 순서와 이름을 맞춰 결합한다.

    Args:
        ex: 영어 운동 코드.
        split: train 또는 validation.
    Returns:
        tuple: 표본, 결합 특징명 목록. 시험 자료는 읽지 않는다.
    """
    rows=[];names=None
    for row in load(ex,split):
        r=row['label'];p=OUT/'image_features'/(r['id']+'.npz')
        with np.load(p) as z:image=z['X'];image_names=z['names'].tolist()
        with np.load(row['feature_path']) as z:_,world_names=compact(z['X'])
        names=['world_'+n for n in world_names]+image_names;rows.append(dict(row,x=np.concatenate([row['x'],image])))
    return rows,names


def train():
    """학습 분할의 교차검증으로 특징 방식을 선택한 후 고정 검증 분할을 평가한다.

    Returns:
        None: image_training_report.json과 모델 파일을 저장한다.
    Raises:
        AssertionError: 참여자 또는 원본이 학습·검증 분할에 겹치는 경우.
    """
    (OUT/'image_gate_policy.json').write_text(json.dumps(POLICY,ensure_ascii=False,indent=2));reports={};bundles={}
    for ex,allowed in [('pushups',['trunk_line','support']),('squats',['descent_range'])]:
        tr,names=expanded(ex,'train');va,_=expanded(ex,'validation')
        for key in ['person','video_sha256']:assert not ({r['label'][key] for r in tr}&{r['label'][key] for r in va})
        items={};models={};columns={}
        for item in allowed:
            pos=tr[0]['label']['criterion_ids'].index(item);a=[r for r in tr if r['label']['criterion_levels'][pos] is not None];b=[r for r in va if r['label']['criterion_levels'][pos] is not None];x=np.array([r['x'] for r in a]);y=np.array([r['label']['criterion_levels'][pos] for r in a]);g=np.array([r['label']['person'] for r in a]);vy=np.array([r['label']['criterion_levels'][pos] for r in b]);vx=np.array([r['x'] for r in b]);counts=collections.Counter(y)
            tokens=CHANNELS['squat_depth' if ex=='squats' else item];variants=[]
            for mode in ['image_only','image_and_world']:
                selected=[i for i,n in enumerate(names) if any(t in n for t in tokens) and (mode!='image_only' or n.startswith('image_'))];oof=np.empty(len(y));baseline=np.empty(len(y))
                for ti,vi in GroupKFold(3).split(x,y,g):
                    reg=fit(x[ti][:,selected],y[ti]);oof[vi]=reg.predict(x[vi][:,selected]);baseline[vi]=np.median(y[ti])
                variants.append(dict(name=mode,columns=selected,feature_count=len(selected),group_MAE_grade=float(mean_absolute_error(y,oof)),group_macro_MAE_grade=float(np.mean([mean_absolute_error(y[y==grade],oof[y==grade]) for grade in set(y)])),group_median_baseline_MAE_grade=float(mean_absolute_error(y,baseline))))
            selected_variant=min(variants,key=lambda v:v['group_macro_MAE_grade']);selected=selected_variant['columns'];reg=fit(x[:,selected],y);pred=reg.predict(vx[:,selected]);vmae=float(mean_absolute_error(vy,pred));base=float(mean_absolute_error(vy,np.full(len(vy),np.median(y))));per_grade={int(grade):dict(samples=int(sum(vy==grade)),MAE=float(mean_absolute_error(vy[vy==grade],pred[vy==grade]))) for grade in sorted(set(vy))};reasons=[]
            if len(counts)<2 or min(counts.values())<POLICY['min_grade_samples']:reasons.append('학습 등급 표본 부족')
            if len(vy)<POLICY['min_validation_samples'] or len(set(vy))<POLICY['min_validation_grades']:reasons.append('검증 등급 다양성 부족')
            if any(v['samples']<POLICY['min_validation_samples_per_grade'] for v in per_grade.values()):reasons.append('등급별 검증 표본 부족')
            if any(v['MAE']>POLICY['max_grade_specific_MAE'] for v in per_grade.values()):reasons.append('오류 등급별 검증 오차 기준 미달')
            if vmae>POLICY['max_validation_MAE_grade']:reasons.append('전체 검증 오차 기준 미달')
            if vmae>base*(1-POLICY['min_relative_baseline_improvement']):reasons.append('중앙값 비교 기준 대비 개선 부족')
            if selected_variant['group_MAE_grade']>=selected_variant['group_median_baseline_MAE_grade']:reasons.append('참여자 교차검증 개선 부족')
            items[item]=dict(status='experimental_candidate' if not reasons else 'validation_failed',web_experiment_enabled=not reasons,hold_reasons=reasons,train_grades={int(k):int(v) for k,v in counts.items()},train_samples=len(a),validation_samples=len(b),validation_grades=dict(collections.Counter(int(v) for v in vy)),validation_MAE_grade=vmae,validation_baseline_MAE_grade=base,validation_per_grade=per_grade,selected_variant=selected_variant['name'],feature_variants=variants,validation_predictions=[dict(id=r['label']['id'],label=int(t),prediction=float(p)) for r,t,p in zip(b,vy,pred)])
            models[item]=reg;columns[item]=selected
        reports[ex]=dict(train_features=len(tr),validation_features=len(va),train_people=len(set(r['label']['person'] for r in tr)),items=items)
        bundles[ex]=dict(models=models,metadata=items,selected_columns=columns,feature_names=names,feature_schema='world98_image112_v5',rubric_version='project-core-form-v1',expert_validated=False,full_score_enabled=False)
    joblib.dump(bundles,OUT/'image_item_models.joblib');result=dict(version='targeted-image-item-model-v5',test_split_used=False,expert_validated=False,production_ready=False,full_score_enabled=False,gate_policy=POLICY,results=reports,limitations=['기존 검증 자료를 반복 사용한 개발 결과. 독립 최종 정확도 아님','특징 방식은 학습 분할의 참여자 교차검증으로 선택','총점 라벨을 생성하거나 미관측 항목을 만점 처리하지 않음'])
    (OUT/'image_training_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':train()
