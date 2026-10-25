"""Evaluate the frozen within-person test after fitting; never select on test."""
import collections,json,sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'error-enrichment'))
import joblib,numpy as np
from train_scores import metrics
from common import sha

B=Path('/home/jayden/SeSACPAI-Mid')
O=Path(os.environ.get('POSTURE_RUN_OUTPUT',str(B/'output/per-person-stratified-80-20-20261024')))

def main():
    model=joblib.load(O/'score-candidates.joblib')
    training=json.loads((O/'score-inputs.frozen.json').read_text())
    test=json.loads((O/'test-inputs.frozen.json').read_text())
    assert model['labels_sha256']==sha(O/'score-inputs.frozen.json')
    trainids={i for h in model['heads'].values() for i in h['trained_ids']}
    assert not trainids & {r['id'] for r in test}
    assert not {s for r in training for s in r['sources']} & {s for r in test for s in r['sources']}
    fingerprint=dict(model_sha256=sha(O/'score-candidates.joblib'),
                     training_inputs_sha256=sha(O/'score-inputs.frozen.json'),
                     test_inputs_sha256=sha(O/'test-inputs.frozen.json'))
    destination=O/'test-evaluation.json'
    if destination.exists():
        previous=json.loads(destination.read_text())
        if previous['frozen_fingerprints']!=fingerprint:
            raise ValueError('Existing test evaluation belongs to another frozen model; do not overwrite.')
        print('Existing frozen evaluation retained');return
    reports=json.loads((O/'score-training.json').read_text())
    assert reports['completed']
    reports={tuple(r['key']):r for r in reports['heads'] if r['status']=='trained_experimental'}
    predictions=[];held=[]
    for row in test:
        available=collections.defaultdict(list)
        targets={}
        for item,target in zip(row['items'],row['levels']):
            targets[item['item']]=target
            key=(row['exercise'],item['item'],item['bank'])
            if target is None or not item['eligible'] or item['feature'] is None or key not in model['heads']:
                continue
            h=model['heads'][key];cv=reports[key]['group_cv']
            if len(item['feature'])!=h['feature_dimensions']:raise ValueError(key)
            qualified=(cv['error_recall'] is not None and cv['error_recall']>=.65 and cv['normal_false_alarm'] is not None and cv['normal_false_alarm']<=.15)
            rank=(-1 if key==('squats','standing_height_return','standing_initial_context_v1') else int(not qualified),cv['balanced_group_MAE'],key[2])
            available[item['item']].append((rank,key,item['feature'],target))
        for criterion,target in targets.items():
            if criterion not in available:
                held.append(dict(id=row['id'],exercise=row['exercise'],criterion=criterion,
                                 reason='label_or_observation_or_trained_head_unavailable'))
                continue
            _,key,x,target=min(available[criterion],key=lambda v:v[0])
            h=model['heads'][key];support=[int(g) for g in h['grade_support']]
            value=float(np.clip(h['model'].predict([x])[0],min(support),max(support)))
            predictions.append(dict(id=row['id'],person=row['person'],group=row['person'],
                exercise=row['exercise'],criterion=criterion,bank=key[2],target=target,prediction=value,
                experimental=True,production_replacement=False))
    by_exercise={exercise:metrics([r for r in predictions if r['exercise']==exercise])
                 for exercise in sorted({r['exercise'] for r in test})}
    by_criterion={exercise+'/'+criterion:metrics([r for r in predictions if r['exercise']==exercise and r['criterion']==criterion])
                  for exercise,criterion in sorted({(r['exercise'],r['criterion']) for r in predictions})}
    result=dict(frozen_fingerprints=fingerprint,head_selection='verified_initial_and_bounded_context_first_when_available_otherwise_Train_CV',
                previously_trained_test_rows_in_new_model=0,source_overlap_with_training=0,
                people_overlap_expected=True,independent_new_person_test=False,
                historical_exposure='Some test samples trained old models; all were previously reviewed. New estimator fits exclude every test source.',
                test_rows=len(test),predicted_item_count=len(predictions),held_item_count=len(held),
                by_exercise=by_exercise,by_criterion=by_criterion,predictions=predictions,held=held,
                test_results_must_not_select_this_run=True,production_replacement=False)
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps(dict(test_rows=len(test),predicted_items=len(predictions),by_exercise=by_exercise),ensure_ascii=False))

if __name__=='__main__':main()
