import json,sys,datetime,copy
from pathlib import Path
import joblib
B=Path(__file__).resolve().parents[2];P=B/'output/return-initial-context-20261024';O=B/'output/return-label-enriched-v4-20261024';S=O/'changed-heads-fit'
sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'tools/paired-evaluation')]
from common import sha
from split_guard import assert_training_allowed,POLICY
from artifact_integrity import equal_state
read=lambda p:json.loads(p.read_text());old=joblib.load(P/'score-candidates.joblib');new=joblib.load(S/'score-candidates.joblib');key=('pushups','return_extension','front');assert set(new['heads'])=={key}
m=copy.deepcopy(old);m['heads'][key]=new['heads'][key];m.update(labels_sha256=sha(O/'score-inputs.frozen.json'),parent_run=str(P),production_replacement=False)
assert all(equal_state(h,m['heads'][k]) for k,h in old['heads'].items() if k!=key)
rows={r['id']:r for r in read(O/'score-inputs.frozen.json')};assert_training_allowed([rows[i] for h in m['heads'].values() for i in h['trained_ids']]);assert not {s for r in rows.values() for s in r['sources']}&{s for r in read(O/'test-inputs.frozen.json') for s in r['sources']}
p=O/'score-candidates.joblib';assert not p.exists();joblib.dump(m,p)
tr=read(P/'score-training.json');nr=read(S/'score-training.json')['heads'][0];tr['heads']=[nr if tuple(h['key'])==key else h for h in tr['heads']];tr.update(completed=True,trained_heads=len(m['heads']),unchanged_heads_copied=len(m['heads'])-1,refitted_heads=1,new_heads=0,gate_passed=sum(h.get('candidate_gate_passed',False) for h in tr['heads']),models_promoted_to_live_service=False)
(O/'score-training.json').write_text(json.dumps(tr,indent=2));(O/'model-freeze-before-test.json').write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_sha256=sha(p),Train_input_sha256=sha(O/'score-inputs.frozen.json'),Test_input_sha256=sha(O/'test-inputs.frozen.json'),policy_sha256=sha(POLICY),labels_sha256=sha(B/'output/return-label-expansion-v3-20261024/labels-v2.frozen.json'),changed_keys=[list(key)],selection_before_Test=True,production_replacement=False),indent=2))
print('58heads; preserved57; refit1',nr['selected'],'CV',nr['group_cv'],'DEV',nr['fixed_development_validation'])
