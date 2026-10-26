"""원본 오류 분류와 AI 초안 점수 회귀를 별도 학습·평가한다. 규칙 점수는 사용하지 않는다."""
import collections
import json
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report,mean_absolute_error,mean_squared_error,r2_score
from common import OUT, save
from features import load_feature


def assert_splits(rows):
    """참여자가 학습·검증·시험 분할에 중복되면 AssertionError를 발생시킨다."""
    persons={s:{r['person'] for r in rows if r['split']==s} for s in ['train','validation','test']}
    assert not persons['train'] & (persons['validation']|persons['test'])
    assert not persons['validation'] & persons['test']


def binary_fit(rows,x,y,destination,source):
    """이진 모델을 검증 자료로 선택하고 최종 시험 결과를 한 번 기록한다.

    Args:
        rows: 분할·참여자 식별자가 포함된 목록.
        x: 관절 특징 행렬. 좌표는 몸통 길이 단위로 정규화되어 있다.
        y: 정상 기준 0, 해당 오류 1의 배열.
        destination: 모델 파일 경로.
        source: 원본 정답 또는 AI 초안이라는 출처 설명.
    Returns:
        자료 부족 사유 또는 검증·시험 지표 객체.
    """
    assert_splits(rows)
    splits=np.array([r['split'] for r in rows]);masks={s:splits==s for s in ['train','validation','test']}
    counts={s:{str(c):int(sum(m & (y==c))) for c in [0,1]} for s,m in masks.items()}
    result=dict(counts=counts,label_source=source)
    if any(min(counts[s].values())<(5 if s=='train' else 3) for s in counts):
        return dict(result,status='insufficient_samples')
    candidates=[]
    for leaf in [1,3,7]:
        model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),ExtraTreesClassifier(n_estimators=160,min_samples_leaf=leaf,class_weight='balanced',random_state=37,n_jobs=4))
        t=time.perf_counter();model.fit(x[masks['train']],y[masks['train']]);fit_seconds=time.perf_counter()-t
        val=classification_report(y[masks['validation']],model.predict(x[masks['validation']]),output_dict=True,zero_division=0)
        candidates.append((val['macro avg']['f1-score'],model,leaf,fit_seconds,val))
    _,model,leaf,fit_seconds,val=max(candidates,key=lambda r:r[0])
    test=classification_report(y[masks['test']],model.predict(x[masks['test']]),output_dict=True,zero_division=0)
    destination.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(dict(model=model,label_source=source,feature_dimensions=x.shape[1],purpose='error_classification_not_numeric_score'),destination)
    return dict(result,status='experimental' if val['macro avg']['f1-score']>=.65 else 'low_validation',validation=val,test=test,selected_leaf=leaf,fit_seconds=fit_seconds,model=str(destination))


def train_errors(items,labels):
    """원본 오류 정답과 AI 세부 오류 초안을 분리하여 모델·평가를 저장한다."""
    report={};cache=OUT/'error_features';labelmap={r['id']:r for r in labels if r.get('eligible_for_experimental_training')}
    for ex in ['pushups','squats','plank','ytw','shoulder']:
        rows=[];values=[]
        for row in items:
            if row['exercise']!=ex:continue
            path=cache/(row['id']+'.npz')
            if not path.exists():continue
            x=load_feature(path)
            if x is not None:rows.append(row);values.append(x)
        if not values:
            report[ex]=dict(status='no_valid_features');continue
        x=np.asarray(values);tasks={}
        errors=sorted({e for r in rows for e in r['errors']})
        for error in errors:
            mask=np.array([r['reference'] or error in r['errors'] for r in rows]);subrows=[r for r,m in zip(rows,mask) if m]
            y=np.array([int(error in r['errors']) for r in subrows]);source='provider_labels' if error!='sequence_order_mismatch' else 'derived_action_order_task_not_provider_posture_error'
            tasks[error]=binary_fit(subrows,x[mask],y,OUT/'models/errors'/ex/(error.replace(' ','_')+'.joblib'),source)
        teacherrows=[(i,r,labelmap[r['id']]) for i,r in enumerate(rows) if r['id'] in labelmap]
        ai={}
        for error in sorted({e for _,_,label in teacherrows for e in label.get('observed_errors',[])}):
            indices=[i for i,_,_ in teacherrows];rs=[r for _,r,_ in teacherrows];y=np.array([int(error in label.get('observed_errors',[])) for _,_,label in teacherrows])
            ai[error]=binary_fit(rs,x[indices],y,OUT/'models/AI_error_drafts'/ex/(error+'.joblib'),'unreviewed_visual_teacher_draft')
        report[ex]=dict(valid_features=len(rows),provider_error_tasks=tasks,AI_error_draft_tasks=ai)
        save(OUT/'error_training_report.json',report)
    return report


def score_gate(rows,y):
    """점수 분포·분할 표본 수가 최소 조건을 만족하는지와 사유를 반환한다."""
    counts=collections.Counter(r['split'] for r in rows)
    if any(counts[s]<n for s,n in [('train',15),('validation',5),('test',5)]):return False,'분할별 유효 점수 라벨 부족'
    train_y=np.asarray([v for r,v in zip(rows,y) if r['split']=='train'])
    if len(set(train_y))<3 or np.ptp(train_y)<10:return False,'학습 점수의 범위 또는 서로 다른 값 부족'
    return True,''


def train_scores(candidates,labels):
    """AI 점수 초안으로 회귀 모델을 학습하되 교사 일치도를 실제 채점 정확도로 부르지 않는다."""
    labelmap={r['id']:r for r in labels if r.get('eligible_for_experimental_training')}
    report={}
    for ex in ['pushups','squats','plank','ytw','shoulder']:
        rows=[];xs=[];ys=[]
        for r in candidates:
            path=OUT/'score_features'/(r['id']+'.npz')
            if r['exercise']!=ex or r['id'] not in labelmap or not path.exists():continue
            x=load_feature(path)
            if x is None:continue
            rows.append(r);xs.append(x);ys.append(labelmap[r['id']]['score'])
        okay,why=score_gate(rows,ys)
        record=dict(label_source='unreviewed_visual_teacher_draft',unit='clip_quality_0_to_100',independent_expert_test=False,
            counts=dict(collections.Counter(r['split'] for r in rows)),score_distribution=dict(collections.Counter(str(v) for v in ys)))
        if not okay:
            report[ex]=dict(record,status='insufficient_score_labels',reason=why);save(OUT/'score_training_report.json',report);continue
        assert_splits(rows);x=np.asarray(xs);y=np.asarray(ys,dtype=float);splits=np.array([r['split'] for r in rows]);tr=splits=='train';va=splits=='validation';te=splits=='test'
        models=[]
        for leaf in [2,4,8]:
            model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),ExtraTreesRegressor(n_estimators=300,min_samples_leaf=leaf,max_features=.7,random_state=37,n_jobs=4))
            start=time.perf_counter();model.fit(x[tr],y[tr]);seconds=time.perf_counter()-start
            models.append((mean_absolute_error(y[va],model.predict(x[va])),model,leaf,seconds))
        val_mae,model,leaf,seconds=min(models,key=lambda v:v[0]);pred=model.predict(x[te]);mean=float(y[tr].mean())
        baseline=mean_absolute_error(y[va],np.full(sum(va),mean))
        path=OUT/'models/scores'/(ex+'.joblib');path.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(dict(model=model,exercise=ex,unit='clip_quality_0_to_100',feature_dimensions=x.shape[1],feature_schema='10fps_32timepoints_normalized_BODY17_xyz_plus_mean_std_min_max',label_source='unreviewed_visual_teacher_draft',expert_validated=False),path)
        predictions=[dict(id=r['id'],split=r['split'],AI_teacher_score=float(truth),model_score=float(p)) for r,truth,p in zip([r for r,m in zip(rows,te) if m],y[te],pred)]
        report[ex]=dict(record,status='experimental_teacher_agreement_only' if val_mae<baseline else 'low_validation_against_mean_baseline',validation_MAE_points=float(val_mae),validation_mean_baseline_MAE_points=float(baseline),test_MAE_points=float(mean_absolute_error(y[te],pred)),test_RMSE_points=float(np.sqrt(mean_squared_error(y[te],pred))),test_R2=float(r2_score(y[te],pred)),selected_leaf=leaf,fit_seconds=seconds,model=str(path),test_predictions=predictions)
        save(OUT/'score_training_report.json',report)
    return report
