"""User-requested within-person 80/20 split; original videos stay whole.

Selection uses only clip counts, exercise names and a fixed hash tie break.
Neither scores nor predictions participate. Historical exposure is retained.
"""
import collections, hashlib, itertools, json, sys
from pathlib import Path

B = Path('/home/jayden/SeSACPAI-Mid')
OLD = B / 'output/error-enrichment-20261024'
NEW = B / 'output/per-person-80-20-20261024'
ANN = B / 'data/r2d2-team-20261023/annotations'
POLICY = ANN / 'per-person-80-20.frozen.json'

def main():
    NEW.mkdir(exist_ok=True)
    labels = json.loads((ANN/'labels.frozen.json').read_text())['labels']
    sources = {r['id']:r for r in json.loads((ANN/'sources.json').read_text())}
    by_source = collections.defaultdict(list)
    for row in labels:
        by_source[row['source_id']].append(row)
    if POLICY.exists():
        policy = json.loads(POLICY.read_text())
    else:
        assignments, summaries, exercise_counts = {}, [], collections.Counter()
        for person in ['P01', 'P02', 'P03', 'P04']:
            videos = sorted(s for s,rs in by_source.items() if rs[0]['person']==person)
            n = sum(len(by_source[s]) for s in videos)
            target = round(n*.2)
            candidates = []
            for k in range(1,len(videos)):
                for subset in itertools.combinations(videos,k):
                    count = sum(len(by_source[s]) for s in subset)
                    if count != target:
                        continue
                    exercises = [by_source[s][0]['exercise'] for s in subset]
                    # Prefer approximately 20% of source videos as well as clips.
                    rank = (abs(k-len(videos)*.2), -len(set(exercises)),
                            sum(exercise_counts[e] for e in set(exercises)),
                            hashlib.sha256(('person80-v1:'+','.join(subset)).encode()).hexdigest())
                    candidates.append((rank,subset))
            if not candidates:
                raise ValueError('Cannot satisfy rounded clip ratio with whole sources: '+person)
            chosen = min(candidates)[1]
            for s in videos:
                assignments[s] = 'test' if s in chosen else 'train'
            exercise_counts.update(by_source[s][0]['exercise'] for s in chosen)
            summaries.append(dict(person=person,clips=n,train_clips=n-target,test_clips=target,
                                  train_sources=len(videos)-len(chosen),test_sources=len(chosen),
                                  test_source_ids=list(chosen),test_fraction=target/n))
        protected_sources = sorted(s for s,role in assignments.items() if role=='test')
        protected_ids = [r['id'] for r in labels if assignments[r['source_id']]=='test']
        policy = dict(schema='within_person_whole_source_80_20_v1',requested_train_fraction=.8,
                      unit='labeled segments rounded within each person; source videos indivisible',
                      assignments=assignments,people=summaries,protected_people=[],
                      protected_clip_ids=protected_ids+['r2d2_'+i for i in protected_ids],
                      protected_source_sha256=[sources[s]['sha256'] for s in protected_sources],
                      test_exercises=sorted(exercise_counts),
                      historical_exposure='Some P01-P03 test clips trained older models; P04 used as development validation. This is a reused-data test, not a fresh independent final test.',
                      selection_used_targets_or_predictions=False,
                      same_person_allowed_across_train_test=True,
                      same_source_allowed_across_train_test=False,
                      final_independent_test_claim_allowed=False)
        POLICY.write_text(json.dumps(policy,ensure_ascii=False,indent=2))
    rows = json.loads((OLD/'score-inputs.frozen.json').read_text())
    originals = {r['id']:r for r in labels}
    training, test = [], []
    for row in rows:
        if row['family']=='R2D2':
            source_id = originals[row['id'][5:]]['source_id']
            row['split'] = policy['assignments'][source_id]
        (test if row['split']=='test' else training).append(row)
    assert not {s for r in training for s in r['sources']} & {s for r in test for s in r['sources']}
    for name, data in [('score-inputs.frozen.json',training),('test-inputs.frozen.json',test)]:
        path = NEW/name
        if path.exists() and json.loads(path.read_text()) != data:
            raise ValueError('Frozen split would change: '+str(path))
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
    # Record actual older training exposure from model receipts, not input roles.
    sys.path.insert(0,str(B/'tools/error-enrichment'))
    import joblib
    history = []
    for folder in [B/'output/integrated-five-20261024',OLD]:
        model = joblib.load(folder/'score-candidates.joblib')
        ids = {i for head in model['heads'].values() for i in head['trained_ids']}
        overlaps = sorted(ids & {r['id'] for r in test})
        history.append(dict(model=str(folder/'score-candidates.joblib'),
                            new_test_ids_previously_trained=overlaps,
                            count=len(overlaps)))
    (NEW/'split-audit.json').write_text(json.dumps(dict(policy_path=str(POLICY),
        people=policy['people'],test_input_rows=len(test),training_and_external_development_rows=len(training),
        test_sources_do_not_enter_training=True,historical_training_exposure=history,
        fresh_independent_test=False,prior_models_not_reused_for_fitting=True),ensure_ascii=False,indent=2))
    print(json.dumps(dict(people=policy['people'],test_input_rows=len(test),
                         historic_exposure_counts=[r['count'] for r in history]),ensure_ascii=False))

if __name__=='__main__':
    main()
