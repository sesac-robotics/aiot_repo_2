"""Anatomical-target ablation: height only, identical corrected timing/labels."""
import json,sys,copy,datetime
from pathlib import Path
B=Path(__file__).resolve().parents[2]; P=B/'output/return-initial-context-20261024';O=B/'output/return-height-only-v1-20261024';O.mkdir(exist_ok=True)
sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import POLICY,assert_training_allowed
read=lambda p:json.loads(p.read_text())
protocol=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_run=str(P),policy_sha256=sha(POLICY),hypothesis='Angle deficits can dominate standing-height severity; compare height-only features after the same verified-initial and bounded-context correction.',new_bank='standing_initial_height_only_v1',feature_indices=[0,1,2,3],feature_names=['hip_height_deficit_mean','hip_height_deficit_max','shoulder_height_deficit_mean','shoulder_height_deficit_max'],no_knee_or_hip_angles=True,labels_unchanged=True,new_failure_labels_in_this_ablation=0,source_roles_unchanged=True,selection='same nested group-CV procedure/C grid/Train-only threshold and fixed gate as parent; no development/Test-driven selection',development_is_previously_exposed=True,independent_final_Test_opened=False,production_replacement=False)
p=O/'MODEL_SELECTION_PROTOCOL.frozen.json';assert not p.exists();p.write_text(json.dumps(protocol,indent=2))
for name in ['score-inputs.frozen.json','test-inputs.frozen.json']:
 rows=read(P/name)
 for r in rows:
  matches=[(i,g) for i,g in zip(r['items'],r['levels']) if i['bank']=='standing_initial_context_v1']
  for i,g in matches:
   new=copy.deepcopy(i);new.update(bank='standing_initial_height_only_v1',feature=i['feature'][:4] if i['feature'] is not None else None,feature_version='standing-initial-height-only-bounded-context-v1');r['items'].append(new);r['levels'].append(g)
 if name.startswith('score'):assert_training_allowed([r for r in rows if r['split']=='train'])
 p=O/name;assert not p.exists();p.write_text(json.dumps(rows,indent=2,allow_nan=False))
(O/'original-test-inputs.frozen.json').write_bytes((P/'test-inputs.frozen.json').read_bytes())
(O/'feature-provenance.json').write_bytes((P/'feature-provenance.json').read_bytes())
source=(B/'tools/return-recheck/train_initial_context.py').read_text().replace("output/return-initial-context-20261024","output/return-height-only-v1-20261024").replace('standing_initial_context_v1','standing_initial_height_only_v1').replace('feature_dimensions=10','feature_dimensions=4')
(B/'tools/return-label-expansion-v3/train_height.py').write_text(source)
evaluate=(B/'tools/return-recheck/evaluate.py').read_text().replace("rank=(-1 if key==('squats','standing_height_return','standing_initial_context_v1') else int(not qualified),cv['balanced_group_MAE'],key[2])","rank=(-2 if key==('squats','standing_height_return','standing_initial_height_only_v1') else -1 if key==('squats','standing_height_return','standing_initial_context_v1') else int(not qualified),cv['balanced_group_MAE'],key[2])")
assert 'rank=(-2' in evaluate
(B/'tools/return-label-expansion-v3/evaluate_height.py').write_text(evaluate)
print('height-only ablation prepared, existing labels/features/roles retained')
