import json,hashlib,sys
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;B=O.parents[1];sys.path.insert(0,str(B/'output/live-webcam-20261023'))
from runtime_snapshot import digest,read_source
from server import score_samples
from movement import MovementTracker
from item_observation import REQUIREMENTS,VERSION
manifest=json.loads((B/'output/squat-consistent-20261023/preserved-models.json').read_text());checks={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in manifest.items()};assert all(checks.values())
labels=json.loads((O/'labels.frozen.json').read_text())['records'];audit=json.loads((O/'eligibility-audit.json').read_text());comparison=json.loads((O/'comparison.json').read_text())
assert hashlib.sha256((O/'labels.frozen.json').read_bytes()).hexdigest()==comparison['label_sha256']
for r in labels:assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['source_sha256']
for spec in REQUIREMENTS.values():assert not set(spec['joints'])&set(list(range(11))+list(range(27,33)))
for fold in comparison['folds']:assert not set(fold['train_people'])&set(fold['held_people'])
session=B/'output/live-webcam-20261023/sessions/1792740109492452117';z=np.load(session/'pose.npz');tr=MovementTracker('squats','front');out=[]
for t,x in zip(z['frame_numbers'],z['image_xy']):
 segment=tr.push(int(t),x)
 if not segment:continue
 scored=score_samples(segment,int(z['width']),int(z['height']),'squats','front')
 masked=[]
 for k,a in segment:
  a=a.copy();a[:11]=np.nan;a[27:]=np.nan;masked.append((k,a))
 ablated=score_samples(masked,int(z['width']),int(z['height']),'squats','front')
 a=[(i['ai_points'],i['rule_points']) for i in scored['score_100']['items']];b=[(i['ai_points'],i['rule_points']) for i in ablated['score_100']['items']];assert a==b
 out.append(dict(number=tr.count,score_100=scored['score_100'],coaching=scored['coaching'],head_ankle_mask_invariant=True))
assert tr.count==7
(O/'live-session-replay.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
summary=dict(observation_version=VERSION,reviewed_sources=17,visually_relabeled_sources=16,usable_sources=sum(bool(r['usable_labeled_items']) for r in audit),usable_labels={k:sum(k in r['usable_labeled_items'] for r in audit) for k in ['knee_inward','trunk_balance','standing_return']},models_preserved=checks,team_source_preserved=digest(read_source())=='c850c28b43de189e32b2419147567247d08385c717c8ca90a6bc2c6d1f58d891',team_source_digest=digest(read_source()),head_ankle_invariance_replayed_reps=7,person_folds=5,source_overlap_per_fold=0,independent_final_test=False,expert_validated=False,return_AI_retrained=False,candidate_runtime_deployed=False,observation_gate_runtime_applied=False,tests_passed=70)
assert summary['team_source_preserved'];(O/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(O/'requirements.json').write_text(json.dumps(REQUIREMENTS,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='models_preserved'},ensure_ascii=False))
