"""학습 분할 안에서만 참여자 단위 교차검증을 수행한다. 시험 자료는 읽지 않는다."""
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from train_direct_scores import model
OUT=Path(__file__).resolve().parent


def check():
    """학습 참여자를 세 묶음으로 나눠 일반화 오차와 단순 비교 기준을 저장한다.

    Returns:
        None: group_validation.json을 저장한다. 오차 단위는 점수 차이(점)이다.
    Raises:
        ValueError: 학습 참여자가 세 명 미만인 경우.
    """
    results={}
    for ex in ['pushups','squats']:
        rows=[];features=[]
        for r in json.loads((OUT/f'{ex}_labels.json').read_text()):
            if r['split']!='train' or not r['eligible_for_experimental_training']:continue
            with np.load(OUT/'features'/(r['id']+'.npz')) as z:
                if z['X'].size:rows.append(r);features.append(z['X'])
        x=np.array(features);y=np.array([r['score'] for r in rows]);groups=np.array([r['person'] for r in rows]);folds=[];predicted=np.zeros(len(y));baseline=np.zeros(len(y))
        for tr,va in GroupKFold(3).split(x,y,groups):
            reg=model(37);reg.fit(x[tr],y[tr]);predicted[va]=reg.predict(x[va]);baseline[va]=y[tr].mean()
            folds.append(dict(held_out_people=sorted(set(groups[va])),samples=len(va),model_MAE=float(mean_absolute_error(y[va],predicted[va])),baseline_MAE=float(mean_absolute_error(y[va],baseline[va]))))
        results[ex]=dict(scope='training_split_only',participants=len(set(groups)),samples=len(rows),folds=folds,pooled_MAE=float(mean_absolute_error(y,predicted)),pooled_baseline_MAE=float(mean_absolute_error(y,baseline)),expert_accuracy=None)
    (OUT/'group_validation.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':check()
