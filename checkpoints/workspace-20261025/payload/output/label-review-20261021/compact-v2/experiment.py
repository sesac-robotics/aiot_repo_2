"""핵심 관절 특징으로 항목별 모델을 학습하고 참여자 교차검증을 비교한다.

기존 시험 집합을 사용하지 않는다. 점수 라벨은 직접 검토한 초안을 재사용하며
각도로 정답 점수를 생성하지 않는다. 시야 밖 항목과 단일 등급은 검증 불가로 남긴다.
"""
import collections
import json
from pathlib import Path
import sys
import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'expanded'
sys.path.insert(0,str(OLD))
from train_direct_scores import model
BODY=[0,11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]
INDEX={k:i for i,k in enumerate(BODY)}


def compact(vector):
    """정규화 관절 시계열에서 작은 특징 벡터를 만든다.

    Args:
        vector: 32시점×17관절×3좌표와 통계가 연결된 기존 특징 배열.
    Returns:
        tuple: 각도(도)·정규화 거리의 요약 배열과 영어 특징명 목록.
        결측이 있는 채널은 NaN과 관측률을 반환한다.
    Raises:
        ValueError: 입력 크기가 예상과 다른 경우.
    """
    seq=vector[:32*17*3].reshape(32,17,3);channels={}
    for name,joints in [('elbow_left',(11,13,15)),('elbow_right',(12,14,16)),('hip_left',(11,23,25)),('hip_right',(12,24,26)),('knee_left',(23,25,27)),('knee_right',(24,26,28)),('body_line_left',(11,23,27)),('body_line_right',(12,24,28)),('shoulder_left',(13,11,23)),('shoulder_right',(14,12,24))]:
        a,b,c=[seq[:,INDEX[j]] for j in joints];u=a-b;v=c-b;den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)
        cosine=np.divide(np.sum(u*v,axis=1),den,out=np.full(32,np.nan),where=den>1e-8)
        channels[name]=np.degrees(np.arccos(np.clip(cosine,-1,1)))
    for name,joints in [('hip_to_ankle_left',(23,27)),('hip_to_ankle_right',(24,28)),('wrist_to_ankle_left',(15,27)),('wrist_to_ankle_right',(16,28))]:
        channels[name]=np.linalg.norm(seq[:,INDEX[joints[0]]]-seq[:,INDEX[joints[1]]],axis=1)
    values=[];names=[]
    for name,channel in channels.items():
        good=channel[np.isfinite(channel)]
        statistics=[*np.percentile(good,[5,25,50,75,95]),float(np.std(good))] if good.size else [np.nan]*6
        for suffix,v in zip(['p05','p25','p50','p75','p95','std','coverage'],statistics+[len(good)/32]):names.append(name+'_'+suffix);values.append(v)
    return np.array(values),names


def experiment():
    """학습 자료 내부의 사람별 검증 결과와 실험 모델을 저장한다.

    Returns:
        None: JSON 보고서와 joblib 모델 파일을 저장한다.
    Raises:
        ValueError: 참여자 또는 유효 표본이 부족한 경우.
    """
    results={};models={}
    for ex,allowed in [('pushups',['trunk_line','descent_range','return_extension','support']),('squats',['descent_range','return_extension'])]:
        rows=[];xs=[]
        for row in json.loads((OLD/f'{ex}_labels.json').read_text()):
            if row['split']!='train':continue
            feature=OLD/'features'/(row['id']+'.npz')
            if not feature.exists():continue
            with np.load(feature) as z:
                if z['X'].size:
                    x,names=compact(z['X']);xs.append(x);rows.append(row)
        x=np.asarray(xs);items={};bundles={}
        for item in allowed:
            index=rows[0]['criterion_ids'].index(item);indices=[i for i,row in enumerate(rows) if row['criterion_levels'][index] is not None]
            y=np.array([rows[i]['criterion_levels'][index] for i in indices],float);groups=np.array([rows[i]['person'] for i in indices]);subx=x[indices];distribution=collections.Counter(int(v) for v in y)
            if len(distribution)<2 or min(distribution.values())<3:
                items[item]=dict(status='insufficient_grade_diversity',grade_counts=dict(distribution),model_trained=False);continue
            predictions=np.empty(len(y));baseline=np.empty(len(y));folds=[]
            for tr,va in GroupKFold(3).split(subx,y,groups):
                reg=model(37);reg.fit(subx[tr],y[tr]);predictions[va]=reg.predict(subx[va]);baseline[va]=y[tr].mean()
                folds.append(dict(validation_people=sorted(set(groups[va])),training_grades=sorted(set(y[tr])),validation_samples=len(va)))
            mae=float(mean_absolute_error(y,predictions));base=float(mean_absolute_error(y,baseline));reg=model(37);reg.fit(subx,y);bundles[item]=reg
            items[item]=dict(status='experimental_only',grade_counts=dict(distribution),model_trained=True,MAE_grade_0_to_2=mae,mean_baseline_MAE_grade_0_to_2=base,MAE_percent_of_item_range=mae*50,folds=folds)
        # 기존 좌표 모델과 같은 표본으로 총점 회귀를 비교한다.
        full=[i for i,r in enumerate(rows) if r['score'] is not None];fy=np.array([rows[i]['score'] for i in full]);fg=np.array([rows[i]['person'] for i in full]);fx=x[full];pred=np.empty(len(full));base=np.empty(len(full))
        for tr,va in GroupKFold(3).split(fx,fy,fg):
            reg=model(37);reg.fit(fx[tr],fy[tr]);pred[va]=reg.predict(fx[va]);base[va]=fy[tr].mean()
        results[ex]=dict(feature_count=len(names),train_samples=len(rows),participants=len(set(r['person'] for r in rows)),item_models=items,compact_total_score_comparison=dict(MAE_points=float(mean_absolute_error(fy,pred)),baseline_MAE_points=float(mean_absolute_error(fy,base)),purpose='기존 좌표 모델과 동일 학습 표본·참여자 분할 비교만 수행. 스쿼트 전체 점수는 시야 불일치 때문에 배포하지 않음.'),excluded_from_single_side_model=[] if ex=='pushups' else ['knee_foot','trunk_balance'])
        models[ex]=dict(item_models=bundles,feature_names=names,expert_validated=False,production_ready=False)
    joblib.dump(models,ROOT/'item_models_experimental.joblib')
    original=json.loads((OLD/'group_validation.json').read_text())
    for ex in results:results[ex]['original_coordinates_MAE_points']=original[ex]['pooled_MAE']
    report=dict(version='compact-kinematics-v2',label_rubric='project-core-form-v1',new_labels_created=0,expert_validated=False,test_split_used=False,selection_scope='training_split_group_cross_validation_only',results=results,limitations=['라벨은 Codex의 장면 검토 초안','여러 모델을 비교한 개발 검증이며 독립 최종 정확도가 아님','충분한 등급 변화가 없는 항목은 학습하지 않음','스쿼트 정면 오류는 측면 단독 모델 대상에서 제외','각도는 MediaPipe 추정 관절에서 계산되며 실제 관절각과 차이가 있을 수 있음'])
    (ROOT/'experiment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':experiment()
