"""지지선 높이 세 축만 사용하는 단조 AI를 학습 그룹 내부에서 선택한다."""
import collections
import copy
import json
import pathlib
import runpy

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from app.scoring.engine import ScoringEngine
from app.scoring.partition import audit_partitions

ROOT=pathlib.Path('/home/jayden/SeSACPAI-Mid/output/five-exercise-recovery-20261023')
OLD=ROOT.parent/'five-exercise-recovery-20261022'


def main() -> None:
    """무릎 각도·무릎/어깨 높이의 물리 단조성만 제한한다. 규칙 정답은 쓰지 않는다."""
    rows=json.loads((ROOT/'plank_shoulder_features.json').read_text())['records'];reserved=json.loads((ROOT/'final_test_hashed_manifest.json').read_text());audit=audit_partitions(rows+reserved['records'],historical_people=reserved['historical_people'],historical_sources=reserved['historical_sources']);groups={key:i for i,g in enumerate(audit['connected_groups']) for key in g['ids']}
    train=[r for r in rows if r['split']=='train' and r['items'][2]['eligible'] and r['levels'][2] is not None and r['items'][2]['bank']=='side_support_clearance_v2'];x=np.asarray([r['items'][2]['feature'] for r in train]);y=np.asarray([r['levels'][2] for r in train]);g=np.asarray([groups[r['id']] for r in train]);all_indices=np.arange(len(y))
    def weights(indices):
        counts=collections.Counter((y[i],g[i]) for i in indices);gc=collections.Counter(a for a,_ in counts)
        return np.asarray([1/(gc[y[i]]*counts[y[i],g[i]]) for i in indices])
    def make(configuration):
        loss,leaf=configuration
        return Pipeline([('physical',ColumnTransformer([('support','passthrough',[1,15,23])])),('model',HistGradientBoostingRegressor(loss=loss,min_samples_leaf=leaf,max_leaf_nodes=5,max_iter=100,l2_regularization=1,monotonic_cst=[1,1,1],random_state=261023))])
    options=[(loss,n) for loss in ['squared_error','absolute_error'] for n in [3,5,10]];losses={}
    for option in options:
        predicted=np.full(len(y),np.nan)
        for a,b in GroupKFold(3).split(x,groups=g):
            model=make(option);model.fit(x[a],y[a],model__sample_weight=weights(a));predicted[b]=model.predict(x[b])
        losses[option]=float(np.average(abs(predicted-y),weights=weights(all_indices)))
    selected=min(options,key=lambda c:losses[c]);model=make(selected);model.fit(x,y,model__sample_weight=weights(all_indices));baseline_model=ScoringEngine().model('plank');previous=('plank','leg_and_body_support','side_support_clearance_v1');key=('plank','leg_and_body_support','side_support_clearance_v2');head=copy.deepcopy(baseline_model.pop(previous));head.update(model=model,records=len(train),people=len(set(g)),grade_counts={str(int(k)):v for k,v in collections.Counter(y).items()},single_grade_model=False,training_only_model_selection=dict(kind='compact_monotone_support',selected=list(selected),selection_MAE={str(k):v for k,v in losses.items()},final_test_used=False));baseline_model[key]=head
    functions=runpy.run_path(str(ROOT/'train_and_compare.py'));report=next(r for r in json.loads((OLD/'held_round2/comparison.json').read_text())['exercises'] if r['exercise']=='plank');baseline={r['id']:r for r in report['fixed_predictions']};predictions=functions['evaluate'](baseline_model,[r for r in rows if r['split']=='validation'],baseline);before=functions['loss'](list(baseline.values()));after=functions['loss'](predictions);adopted=after['AI_MAE']<=before['AI_MAE']-1 and after['error_item_MAE']<=before['error_item_MAE']+1e-10
    result=dict(baseline=before,candidate=after,adopted=adopted,selected=list(selected),selection_MAE={str(k):v for k,v in losses.items()},fixed_predictions=predictions,final_test_opened=False,validation_is_reused_development=True,rule_unchanged=True)
    (ROOT/'plank_compact_trial.json').write_text(json.dumps(result,indent=2));joblib.dump(baseline_model,ROOT/'models/plank_compact_candidate.joblib',compress=3);print(before,after,adopted)


if __name__=='__main__':main()
