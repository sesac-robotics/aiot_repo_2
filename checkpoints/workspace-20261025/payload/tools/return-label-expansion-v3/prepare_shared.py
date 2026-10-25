"""Pool the same angle/visibility measurement across views, without label copying."""
import json,sys,copy,datetime
from pathlib import Path
B=Path(__file__).resolve().parents[2];P=B/'output/return-label-enriched-v4-20261024';O=B/'output/return-shared-extension-v1-20261024';O.mkdir(exist_ok=True);S=O/'changed-heads-fit';S.mkdir(exist_ok=True)
sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import POLICY,assert_training_allowed
from shared_features import pool_item
read=lambda p:json.loads(p.read_text());bank='observed_arm_extension_shared_v1';audit=[];counts={}
for name in ['score-inputs.frozen.json','test-inputs.frozen.json']:
 rows=read(P/name)
 for r in rows:
  if r['exercise']!='pushups':continue
  possible=[(i,g) for i,g in zip(r['items'],r['levels']) if i['item']=='return_extension' and i['bank'] in ['front','side_legacy_roi_v1']]
  if not possible:continue
  assert len(possible)==1, r['id']
  i,g=possible[0];f=i['feature'];assert f is None or len(f)==8
  # Both source banks call the same three_items.describe('pushups') routine:
  # coherent shoulder/elbow/wrist angle, terminal [5,10,25,50,75,90,95] percentiles + visibility.
  a=pool_item(i);r['items'].append(a);r['levels'].append(g)
  audit.append(dict(id=r['id'],split=r['split'],source_bank=i['bank'],target=g,eligible=i['eligible'],feature_identical=True))
 if name.startswith('score'):assert_training_allowed([r for r in rows if r['split']=='train']);training=rows
 p=O/name;assert not p.exists();p.write_text(json.dumps(rows,indent=2,allow_nan=False));counts[name]=len(rows)
subset=[]
for r in training:
 pairs=[(i,g) for i,g in zip(r['items'],r['levels']) if i['bank']==bank]
 if pairs:
  a=copy.deepcopy(r);a['items']=[i for i,g in pairs];a['levels']=[g for i,g in pairs];subset.append(a)
(S/'score-inputs.frozen.json').write_text(json.dumps(subset,indent=2,allow_nan=False));(O/'feature-provenance.json').write_text(json.dumps(audit,indent=2));(O/'original-test-inputs.frozen.json').write_bytes((P/'test-inputs.frozen.json').read_bytes())
(O/'MODEL_SELECTION_PROTOCOL.frozen.json').write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_run=str(P),policy_sha256=sha(POLICY),bank=bank,measurement='identical three_items.describe pushup return_extension angle quantiles and observation fraction, across front and side_legacy_roi_v1',source_views_not_claimed_interchangeable_accuracy=True,new_manual_labels=0,selection='unchanged Train group CV/generic model grid, no Test tuning; existing development gate',source_roles_and_labels_unchanged=True,Test_original_features_preserved=True,Test_features_augmented_without_new_targets=True,independent_final_Test_opened=False,production_replacement=False),indent=2))
e=(B/'tools/return-recheck/evaluate.py').read_text().replace("rank=(-1 if key==('squats','standing_height_return','standing_initial_context_v1') else int(not qualified),cv['balanced_group_MAE'],key[2])","rank=(-1 if key==('squats','standing_height_return','standing_initial_context_v1') else int(not qualified),cv['balanced_group_MAE'],key[2])")
# Shared bank competes solely by existing Train-CV rank; never force an inferior view model.
(B/'tools/return-label-expansion-v3/evaluate_shared.py').write_text(e)
print('Shared same-measurement bank appended',counts,'training+development items',len(subset))
