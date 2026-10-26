import json,sys
from pathlib import Path
B=Path(__file__).resolve().parents[2];O=B/'output/mild-return-mining-v2-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
read=lambda p:json.loads(p.read_text());items=[]
for r in read(O/'intake.frozen.json'):
 items.append(dict(id=r['id'],person=r['person'],split='train',source_sha256=r['source_sha256'],original_grade=r['old_grade'],revised_grade=r['old_grade'],reviewer='AI_assisted_visual_draft_not_expert',decision='no_unambiguous_new_partial_return_evidence; original label retained',sheet=f"review-{(int(r['review_id'][1:])-1)//4+1:02}.jpg"))
for folder in ['qevd','qevd-more']:
 for r in read(O/folder/'intake.frozen.json'):
  note='Observed completed returns or no clear partial failure; not a new error label. Truncated final descent does not prove a return failure.'
  if folder=='qevd-more' and r['review_id']=='Q11':note='Focused dense review: starts standing, then stays lowered; no clear completed upper reversal and next descent. Hold/incomplete not converted to grade0 or1.'
  if folder=='qevd-more' and r['review_id']=='Q09':note='Dense review resolves suspected low upper frames: actually stands at120/123 before the next descent; rejected as a partial-return error.'
  items.append(dict(id=r['id'],person=r['person'],split='train',source_sha256=r['source_sha256'],standing_height_grade=None,fit_eligible=False,decision=note,reviewer='AI_assisted_visual_draft_not_expert',sheet=folder+'/'+r['review_id']+'-01.jpg'))
for r in read(O/'multiview/intake.frozen.json'):
 note='Provider bad label does not equal a return error; sampled repeated upper positions do not establish a new initial-height return failure. No new severity label.'
 if r['review_id']=='M01':note='Repeated visibly incomplete hip/knee extension is an error candidate, but the clip begins bent and never supplies a verified full-standing reference. Standing-height grade remains unknown; not fit as an initial-height-return example.'
 items.append(dict(id=r['id'],person=r['person'],split='train',source_sha256=r['source_sha256'],standing_height_grade=None,fit_eligible=False,decision=note,observed_fault_candidate='repeated_partial_hip_knee_extension' if r['review_id']=='M01' else None,reference_status='verified_initial_standing_not_available' if r['review_id']=='M01' else 'needs_dense_reference_target_confirmation',reviewer='AI_assisted_visual_draft_not_expert'))
unknown=dict(id='squat_front_m4_6_6_r07_to_r10',split='train',person='cfrep_m4',standing_height_grade=None,decision='Additional dense review of frames268–436 against41 does not establish a height-return failure just from looking downward/leaning; original unassigned labels retained.',sheet='m4-unassigned.jpg')
result=dict(Train_only=True,independent_final_Test_opened=False,new_failure_grade_labels=0,new_failure_fit_rows=0,source_roles_unchanged=True,reviewed=items,additional_unassigned_review=unknown,viewed_sheets_sha256={str(p.relative_to(O)):sha(p) for p in sorted(O.glob('**/*.jpg'))},new_training_expansion_not_claimed=True)
assert len(items)==66;(O/'review-decisions.frozen.json').write_text(json.dumps(result,indent=2));dest=B/'data/return-discrimination-review-20261024';dest.mkdir(exist_ok=True);(dest/'review-decisions.frozen.json').write_bytes((O/'review-decisions.frozen.json').read_bytes());print('REVIEW_FROZEN66records:38intervals+28sourcevideos; no fabricated failure grades')
