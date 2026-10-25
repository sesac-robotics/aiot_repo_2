import datetime,json,sys
from pathlib import Path
import joblib
B=Path(__file__).resolve().parents[2];OLD=B/'output/return-root-cause-v3-20261024';O=B/'output/mild-return-calibrated-20261024';SUB=O/'changed-heads-fit';sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'tools/paired-evaluation')]
from common import sha
from split_guard import assert_training_allowed
from artifact_integrity import equal_state
read=lambda p:json.loads(p.read_text());old=joblib.load(OLD/'score-candidates.joblib');new=joblib.load(SUB/'score-candidates.joblib');changed=set(new['heads']);assert len(changed)==3 and changed<=old['heads'].keys()
models=dict(old,heads={**old['heads'],**new['heads']},labels_sha256=sha(O/'score-inputs.frozen.json'),parent_run=str(OLD),changed_heads=sorted(changed));assert sum(equal_state(h,old['heads'][k]) for k,h in models['heads'].items() if k not in changed)==53
rows=read(O/'score-inputs.frozen.json');byid={r['id']:r for r in rows};assert_training_allowed([byid[i] for h in models['heads'].values() for i in h['trained_ids']]);assert sha(O/'test-inputs.frozen.json')==sha(OLD/'test-inputs.frozen.json')
p=O/'score-candidates.joblib';assert not p.exists();joblib.dump(models,p)
report=read(OLD/'score-training.json');rh={tuple(h['key']):h for h in report['heads']};rh.update({tuple(h['key']):h for h in read(SUB/'score-training.json')['heads']});report.update(heads=list(rh.values()),completed=True,trained_heads=len(models['heads']),gate_passed=sum(h.get('candidate_gate_passed',False) for h in rh.values()),refitted_heads=3,unchanged_heads_copied=53,models_promoted_to_live_service=False)
(O/'score-training.json').write_text(json.dumps(report,indent=2));(O/'model-freeze-before-test.json').write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_sha256=sha(p),train_sha256=sha(O/'score-inputs.frozen.json'),test_sha256=sha(O/'test-inputs.frozen.json'),unchanged_heads=53,changed_heads=sorted(changed),production_replacement=False),indent=2));print('MERGED56; preserved53')
