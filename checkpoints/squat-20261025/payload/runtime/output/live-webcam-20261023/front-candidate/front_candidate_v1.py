"""기존 모델과 분리한 정면 하강 범위 추정·화면 복귀 점수 후보."""
from pathlib import Path
import numpy as np
import joblib
MODEL=Path(__file__).parent/'front-candidate/model.joblib'
VERSION='front-squat-observable70-experimental-v1'

def vector(xy):
    """픽셀 단위에 무관한 정면 특징을 만든다. 발목은 요구하지 않는다."""
    x=np.asarray(xy,float)
    torso_ok=np.isfinite(x[:,[11,12,23,24]]).all((1,2))
    knee_ok=np.isfinite(x[:,[25,26]]).all(2)
    valid=torso_ok & knee_ok.any(1)
    if torso_ok.mean()<.8 or valid.mean()<.8 or valid.sum()<10:raise ValueError('어깨·골반·한쪽 이상 무릎 관측 부족')
    a=x[valid];s=a[:,[11,12]].mean(1);h=a[:,[23,24]].mean(1)
    # 양 무릎이 보이면 평균, 하나만 보이면 관측된 무릎 높이만 사용한다.
    k=np.nanmean(a[:,[25,26]],axis=1)
    t=x[torso_ok];ts=t[:,[11,12]].mean(1);th=t[:,[23,24]].mean(1)
    scale=float(np.median(np.linalg.norm(th-ts,axis=1)))
    if scale<1:raise ValueError('몸통 길이 부족')
    hy=th[:,1];dy=(hy-np.median(hy[:3]))/scale
    hk=(k[:,1]-h[:,1])/scale
    # 추정·채움 없이 관측률이 높은 한쪽 허벅지 투영을 사용한다.
    side=0 if knee_ok[:,0].sum()>=knee_ok[:,1].sum() else 1
    thigh=x[valid,25+side]-x[valid,23+side]
    proj=thigh[:,1]/np.maximum(1,np.linalg.norm(thigh,axis=1));proj=proj[np.isfinite(proj)]
    if len(proj)<10:raise ValueError('관측 허벅지 부족')
    tilt=np.degrees(np.arctan2(th[:,0]-ts[:,0],th[:,1]-ts[:,1]))
    return np.array([np.percentile(dy,95)-np.percentile(dy,5),np.percentile(dy,95),np.percentile(hk,10),np.percentile(hk,50),np.percentile(proj,10),np.percentile(proj,50),np.percentile(np.abs(tilt),90),abs(np.median(hy[-3:])-np.median(hy[:3]))/scale])


def rule_depth(value,bounds):
    low,high=bounds
    return 2 if value>=high else 1 if value>=low else 0

def predict(xy):
    """학습 AI, 독립 규칙, 혼합 점수를 이름을 구분해 반환한다."""
    bundle=joblib.load(MODEL);v=vector(xy)
    ai=float(np.clip(bundle['model'].predict([v])[0],0,2))
    rule=rule_depth(v[bundle['rule_feature']],bundle['rule_boundaries'])
    # 화면상 복귀 높이 정책은 학습된 신전 각도 판정이 아니다.
    ret=2 if v[7]<=.08 else 1 if v[7]<=.20 else 0
    trunk=2 if v[6]<=8 else 1 if v[6]<=15 else 0
    common=20*ret/2+20*trunk/2
    return dict(version=VERSION,experimental=True,ai_depth_grade=round(ai,3),rule_depth_grade=rule,
        return_height_grade=ret,trunk_grade=trunk,ai_depth_points=round(30*ai/2,1),
        rule_points=round(30*rule/2+common,1),ai_assisted_points=round(30*ai/2+common,1),maximum=70,
        pure_ai_total=None,features=v.tolist(),
        methods={'depth_ai':'기존 동기화 측면 시각 라벨로 학습한 정면 입력 회귀', 'depth_rule':'같은 학습 라벨로 독립 보정한 단일 특징 경계','return':'화면 복귀 높이 차/몸통 길이 ≤0.08/0.20의 실험 정책','trunk':'기존 좌우 기울기 8/15도 경계'},
        limitations=['정면 관측 점수이며 기존 100점 총점과 다름','AI 보조 합계는 AI 하강+규칙 복귀·몸통이며 순수 AI 총점이 아님','복귀 규칙 오류 등급 검증 부족','몸통과 한쪽 이상 무릎의 동시 관측 80% 필요; 양쪽 무릎 대칭 점수는 계산하지 않음'])
