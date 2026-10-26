"""관절 특징에서 검증 기준을 통과한 항목만 모델로 예측한다. 종합 점수는 만들지 않는다."""
import hashlib
import json
from pathlib import Path
import sys
import joblib
import numpy as np
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'compact-v2'))
from experiment import compact,INDEX
WEIGHTS={'pushups':{'trunk_line':40,'descent_range':30,'return_extension':20,'support':10},'squats':{'descent_range':30,'return_extension':20}}
NAMES={'trunk_line':'몸통·골반 정렬','descent_range':'하강 범위','return_extension':'상단 복귀','support':'손·발 지지'}


def predict(vector,exercise,artifact=None,image_vector=None,diagnostics=False):
    """학습 모델을 사용해 항목별 실험 점수와 보류 근거를 반환한다.

    Args:
        vector: 기존 Heavy 관절 특징 벡터. 32시점×17관절×3좌표를 포함한다.
        exercise: pushups 또는 squats.
        artifact: 검사 시 주입할 저장 모델 사전. None이면 활성 모델 파일을 읽는다.
        image_vector: 활성 모델에 필요한 112차원 화면 기하 특징. 없으면 항목을 보류한다.
        diagnostics: True이면 관측 가능한 미검증 항목의 등급 예측도 별도 반환한다.
    Returns:
        dict: 항목별 예측 등급(0~2), 배점 단위 점수, 보류 이유.
        total_score는 항상 None이다. 관절 관측률은 확률·정확도가 아니다.
    Raises:
        ValueError: 운동 코드·특징 크기·모델 출력이 잘못된 경우.
    """
    if exercise not in WEIGHTS or len(vector)!=1836:raise ValueError('운동 코드 또는 특징 형식 오류')
    config=json.loads((OUT/'active_model.json').read_text()) if (OUT/'active_model.json').exists() else dict(file='item_models.joblib',version='targeted-item-model-v4-balanced')
    model_path=OUT/config['file']
    artifact=joblib.load(model_path) if artifact is None else artifact
    bundle=artifact[exercise];x,names=compact(np.asarray(vector));seq=np.asarray(vector)[:1632].reshape(32,17,3);items=[]
    # 양쪽 중 한쪽에서라도 주요 관절이 같은 시점에 관측되어야 한다.
    joint_groups=[(11,23,25,27),(12,24,26,28)]
    visible=max(float(np.isfinite(seq[:,[INDEX[k] for k in group],:]).all(axis=(1,2)).mean()) for group in joint_groups)
    for item,weight in WEIGHTS[exercise].items():
        meta=bundle['metadata'].get(item,dict(web_experiment_enabled=False,hold_reasons=['학습 등급 자료 부족']));record=dict(id=item,name_ko=NAMES[item],weight=weight,predicted_grade=None,predicted_points=None,diagnostic_grade=None,status='withheld',reasons=[])
        enabled=meta.get('web_experiment_enabled',False)
        if not enabled:record['reasons']=list(meta.get('hold_reasons') or ['학습 등급 자료 부족'])
        if visible<.7:record['reasons'].append('어깨·골반·무릎·발목이 함께 보이는 구간 부족')
        elif 'selected_columns' in bundle and image_vector is None:record['reasons'].append('화면 기하 특징이 없어 모델 예측을 보류함')
        elif item in bundle['models'] and (enabled or diagnostics):
            input_features=x
            if 'selected_columns' in bundle:
                if len(image_vector)!=112:raise ValueError('화면 기하 특징 크기 오류')
                input_features=np.concatenate([x,image_vector])[bundle['selected_columns'][item]]
            grade=float(bundle['models'][item].predict(input_features[None,:])[0])
            if not np.isfinite(grade) or not 0<=grade<=2:raise ValueError('비정상 모델 출력')
            if enabled:record.update(predicted_grade=grade,predicted_points=grade*weight/2,status='experimental_prediction')
            elif diagnostics:record['diagnostic_grade']=grade
        items.append(record)
    return dict(exercise=exercise,items=items,total_score=None,full_score_enabled=False,expert_validated=False,feature_visibility_fraction=visible,model_version=config['version'],model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),score_notice='관측 가능한 개별 항목의 실험 예측입니다. 종합 100점 점수가 아닙니다.')
