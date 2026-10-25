"""Add visually reviewed native-Train attempts, retaining frozen team Test targets."""
import json,sys,copy,collections
from pathlib import Path
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/label-verified-five-expanded-20261024';BASE=B/'output/label-verified-five-20261024'
sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'tools/integrated-five'),str(B/'five-exercise-comparison')]
from common import sha
from pose_features import arm_pose
from team_features import describe
from recovery_features import describe_ytw
from features import shape
from split_guard import assert_training_allowed
rows=json.loads((BASE/'score-inputs.frozen.json').read_text());new=json.loads((O/'new-labels.frozen.json').read_text());existing={r['person']:r['group'] for r in rows};provenance=[]
for d in new:
 p=O/'observations'/d['review_id']/'primary.npz';z=np.load(p);times=z['frame_numbers']/d['fps'];seq=z['image_xy'][(times>=d['start'])&(times<d['end'])];v=d['variant']
 if v=='upright':items=describe('ytw',seq,'front',d['end']-d['start'])
 else:
  items=describe_ytw(seq)
  for item in items:item.update(bank='front',holds=[])
 for item in items:
  for k in ['rule_grade','measurement','higher_is_better','proxy_only']:item.pop(k,None)
 levels=d['grades'].copy()
 # Missing phases are labels of absence, not missing pose: pool the complete observed attempt.
 for i,kind in enumerate(['Y','T','W']):
  for bank,f in [(v+'_arm_pose_v2',arm_pose(seq)),(v+'_compact_shape_v3',shape(seq,kind))]:
   eligible=f is not None and levels[i] is not None
   items.append(dict(item=items[i]['item'],bank=bank,feature=f if eligible else None,eligible=eligible,holds=[] if eligible else ['manual_grade_or_required_pose_unobserved'],features_contain_no_rule_grades=True,eligibility_contract='manual_complete_attempt_and_80_percent_required_arm_visibility'))
   levels.append(d['grades'][i])
 for item,g in zip(items,levels):
  if g is None:item.update(eligible=False,feature=None,holds=['visual_grade_uncertain'])
 row=dict(id=d['id'],exercise='ytw',person=d['person'],group=existing.get(d['person'],d['person']),family='QEVD',origin='label_recheck_enrichment',split='train',sources=d['sources'],recording_groups=[],levels=levels,items=items,reference_total_allowed=all(g is not None for g in d['grades']),whole_attempt_observed=True,view='front',variant=v,annotation_review=d['annotation_status'])
 assert_training_allowed([row]);rows.append(row);provenance.append(dict(id=d['id'],cache=str(p),cache_sha256=sha(p),source_sha256=d['sources'][0],start=d['start'],end=d['end'],observed_frames=len(seq)))
# Shared room across different provider IDs is not treated as independent CV evidence.
alias={'qevd_1948','qevd_370'}
assert all(r['split']=='train' for r in rows if r['person'] in alias)
for r in rows:
 if r['person'] in alias:r['group']='capture_shared_rack_1948_370'
test=json.loads((BASE/'test-inputs.frozen.json').read_text());assert not {s for r in rows for s in r['sources']}&{s for r in test for s in r['sources']}
assert len({r['id'] for r in rows})==len(rows)
for name,value in [('score-inputs.frozen.json',rows),('test-inputs.frozen.json',test),('new-feature-provenance.json',provenance)]:
 p=O/name;assert not p.exists() or json.loads(p.read_text())==value;p.write_text(json.dumps(value,indent=2,allow_nan=False))
(O/'preparation.json').write_text(json.dumps(dict(base=str(BASE),total_rows=len(rows)+len(test),training_external_validation=len(rows),test_rows=len(test),new_segments=len(new),new_sources=len({s for r in new for s in r['sources']}),test_inputs_identical_sha256=sha(BASE/'test-inputs.frozen.json')==sha(O/'test-inputs.frozen.json'),test_semantically_unchanged=test==json.loads((BASE/'test-inputs.frozen.json').read_text()),source_assignments_unchanged=True,capture_group_alias=sorted(alias)),indent=2))
print('EXPANDED',len(rows),len(test),len(new),flush=True)
