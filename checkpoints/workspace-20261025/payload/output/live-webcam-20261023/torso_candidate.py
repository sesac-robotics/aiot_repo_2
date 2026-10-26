"""발목·무릎 누락과 독립적인 정면 복귀·몸통 특징 후보."""
import numpy as np

def torso_features(xy):
    """상단 구간의 강건 높이와 몸통·어깨 기울기를 측정한다."""
    x=np.asarray(xy,float);ok=np.isfinite(x[:,[11,12,23,24]]).all((1,2))
    if ok.mean()<.8 or ok.sum()<10:raise ValueError('몸통 관측 부족')
    t=x[ok];s=t[:,[11,12]].mean(1);h=t[:,[23,24]].mean(1)
    scale=np.median(np.linalg.norm(h-s,axis=1))
    if scale<1:raise ValueError('몸통 길이 부족')
    y=h[:,1];n=min(10,max(3,len(y)//4))
    # 준비·복귀 끝의 한 프레임 대신 상단에 가까운 세 관측의 중앙값.
    start=np.median(np.sort(y[:n])[:3]);end=np.median(np.sort(y[-n:])[:3])
    return_delta=abs(end-start)/scale
    axis=np.degrees(np.arctan2(h[:,0]-s[:,0],h[:,1]-s[:,1]))
    shoulder=np.degrees(np.arctan2(t[:,12,1]-t[:,11,1],np.abs(t[:,12,0]-t[:,11,0])))
    # 수평 원점·크기에 불변인 좌우 기울기와 어깨선 기울기.
    return np.array([return_delta,np.percentile(abs(axis),90),np.percentile(abs(shoulder),90),np.percentile(abs(axis-np.median(axis[:3])),90)])
