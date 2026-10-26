"""Versioned observation safeguards around preserved candidate models; no retraining on this session."""
from pathlib import Path
from functools import lru_cache
import hashlib
import importlib.util
import joblib
import numpy as np
ROOT=Path(__file__).resolve().parent.parent/'squat-relabel-expanded-20261023'
@lru_cache(maxsize=1)
def load():
    report=__import__('json').loads((ROOT/'report.json').read_text())
    assert hashlib.sha256((ROOT/'candidate.joblib').read_bytes()).hexdigest()==report['model_sha256']
    bundle=joblib.load(ROOT/'candidate.joblib')
    assert hashlib.sha256((ROOT/'labels.frozen.json').read_bytes()).hexdigest()==bundle['label_sha256']
    spec=importlib.util.spec_from_file_location('accepted_criteria_features',ROOT/'criteria_features.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return bundle['heads'],module.observed_features

def evaluate(xy,width=None,height=None):
    heads,features=load();x=np.asarray(xy,float);guarded=x.copy();feet=[29,30,31,32]
    finite=np.isfinite(x[:,feet]).all(2)
    edge=np.zeros_like(finite)
    if width and height:
        a=x[:,feet]
        edge=finite & ((a[:,:,0]<width*.02)|(a[:,:,0]>width*.98)|(a[:,:,1]<height*.02)|(a[:,:,1]>height*.98))
        outside=finite & ((a[:,:,0]<0)|(a[:,:,0]>=width)|(a[:,:,1]<0)|(a[:,:,1]>=height))
        for j,k in enumerate(feet):guarded[outside[:,j],k]=np.nan
    f=features(guarded);out=[]
    for key,method,title in [('trunk_balance','ai','몸통 좌우 기울기'),('heel_support','rule','뒤꿈치·발 지지')]:
        obs=f.get(key);item=dict(id=key,method=method,title=title,grade=None,maximum=2,status='미관측',coverage=0,experimental=True)
        if obs:
            h=heads[key]
            if method=='ai':grade=float(np.clip(h['model'].predict([obs['x']])[0],0,2))
            else:
                good,partial=h['boundaries'];v=obs['scalar'];grade=2 if v<=good else 1 if v<=partial else 0
            item.update(grade=round(grade,3),status='검증 중인 항목 평가',coverage=obs['coverage'])
        if key=='trunk_balance':
            if obs:
                torso=x[np.isfinite(x[:,[11,12,23,24]]).all((1,2))]
                sh=torso[:,[11,12]].mean(1);hip=torso[:,[23,24]].mean(1)
                tilt=np.degrees(np.arctan2(hip[:,0]-sh[:,0],hip[:,1]-sh[:,1]));roll=np.degrees(np.arctan2(torso[:,12,1]-torso[:,11,1],abs(torso[:,12,0]-torso[:,11,0])))
                # Same 10-degree cue as live coaching; both features measured independently of AI.
                td=float(np.percentile(abs(tilt-np.median(tilt[:3])),90));rd=float(np.percentile(abs(roll-np.median(roll[:3])),90))
                signal=max(td,rd);level=2 if signal<10 else 1 if signal<15 else 0
                disagreement=bool(grade<1.5 and level==2 or grade>=1.5 and level==0)
                item.update(observation_grade=level,tilt_change_degrees=round(td,2),shoulder_change_degrees=round(rd,2),disagreement=disagreement,guidance='큰 좌우 기울임 신호 없음 · AI 점수와 차이 있어 감점 원인 미확정' if disagreement and level==2 else '큰 좌우 기울임 신호 없음' if level==2 else '몸통·어깨의 좌우 기울임 변화 관측 · 정면 유지 확인',grade_meaning='AI 연속 등급은 10도 경고 규칙과 동일 기준이 아님')
        else:
            usable=np.isfinite(guarded[:,feet]).all(2);coverage=float(usable.all(1).mean())
            item.update(coverage=coverage,raw_coverage=float(finite.all(1).mean()),edge_fraction=float(edge.any(1).mean()),left_coverage=float(usable[:,[0,2]].all(1).mean()),right_coverage=float(usable[:,[1,3]].all(1).mean()),reason='frame_edge' if edge.any() and obs is None else 'missing_heel_or_toe' if obs is None else None,guidance='발이 화면 아래·옆 경계에 걸림 · 카메라 구도 조정 필요' if edge.any() and obs is None else '뒤꿈치·발끝 관측 부족 · 발 지지만 미평가' if obs is None else '관측 범위에서 발 지지 유지 추정' if grade==2 else '발 지지 변화 관측 · 뒤꿈치 접촉 확인')
        out.append(item)
    return dict(version='checklist-selective-v2-observation-guard',items=out,total=None,scope='몸통 AI·발 지지 규칙; 경계·관측 진단과 AI/규칙 차이를 구분',validation='모델: 1명 10회 개발 검증. 새 관측 보호 로직은 저장 자료 재실행 검증')
