import sys,collections,cv2
from fractions import Fraction
import numpy as np
from common import B,O,read,save,sha,status
sys.path.insert(0,str(B/'five-exercise-comparison'));import recovery_train,three_items
from team_features import describe,height_features

def main():
 old=B/'output/five-exercise-recovery-20261022';rows=read(old/'held_round2/candidate_input_manifest.json');rows=[r for r in rows if r['exercise']!='squats']+read(old/'squat_partial_top_trial/candidate_input_manifest.json');rows+=read(B/'output/five-exercise-recovery-20261023/new_feature_rows.json')['records']
 for r in rows:
  r['origin']='legacy';r['family']='QEVD' if r['id'].startswith('qevd') else 'REHAB24' if r['id'].startswith('rehab24') else 'MultiView' if r['id'].startswith('subject') else 'US_Navy' if 'navy' in r['id'] or '773312' in r['id'] else 'other_reviewed';r['label_origin']='direct_visual_draft'
 inventory={r['id']:r for r in read(B/'output/r2d2-review-20261024/inventory.json')};labels=read(B/'data/r2d2-team-20261023/annotations/labels.frozen.json')['labels'];excluded=[];team=[]
 for r in labels:
  if not r['quality_training_candidate']:excluded.append(dict(id=r['id'],reason='incomplete_cycle_or_different_support_variant'));continue
  src=inventory[r['source_id']];cap=cv2.VideoCapture(src['copy']);width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release();z=np.load(O/'team-observations'/r['source_id']/'side.npz');frames=z['frame_numbers'];mask=(frames/float(Fraction(src['fps']))>=r['start'])&(frames/float(Fraction(src['fps']))<r['end']);seq=three_items.mask_image_border(z['image_xy'][mask],width,height);baseline=height_features.reference(z['image_xy'][:8]);notes=describe(r['exercise'],seq,r['view'],r['end']-r['start'],baseline,width,height);levels=[i['grade'] for i in r['items']]
  # Semantic masks: side-only knee-extension is not standing-height return.
  if r['exercise']=='squats' and r['view']=='side':levels[1]=None
  team.append(dict(id='r2d2_'+r['id'],exercise=r['exercise'],person='r2d2_'+r['person'],sources=[r['source_sha256']],recording_groups=['r2d2_'+r['person']],split='validation' if r['person']=='P04' else 'train',levels=levels,items=notes,origin='R2D2',family='R2D2',view=r['view'],variant=r['variant'],label_origin='direct_visual_draft',reference_total_allowed=r['total_score'] is not None,whole_attempt_observed=True))
  for i,n in enumerate(notes):
   if levels[i] is None:n['eligible']=False
   if n['feature'] is None or not np.isfinite(n['feature']).all():n['eligible']=False;n['feature']=None
 rows+=team
 # Previously reviewed partial-return clips are real source videos, not synthetic errors.
 prior=read(B/'output/squat-return-20261023/feature-rows.json');prlabels={r['id']:r for r in read(B/'output/squat-return-20261023/labels.frozen.json')['labels']}
 for r in prior:
  lab=prlabels[r['id']];source=sha(lab['path']);note=dict(item='standing_height_return',bank='height_v4',eligible=True,feature=r['features']['x'],holds=[])
  rows.append(dict(id='return_'+r['id'],exercise='squats',person=r['person'],sources=[source],recording_groups=[r['person']],split=r['split'],levels=[None,r['target'],None,None],items=[dict(item='not_used',bank='none',eligible=False,feature=None),note,dict(item='not_used',bank='none',eligible=False,feature=None),dict(item='not_used',bank='none',eligible=False,feature=None)],origin='prior_return_review',family='CFRep_reviewed',view=r['view'],label_origin='direct_visual_draft',reference_total_allowed=False))
 views={r['source_id']:r['view'] for r in labels};save('extraction-view-audit.json',[dict(source_id=r['id'],declared_video_view=views[r['id']],storage_slot='side.npz',training_uses_declared_video_view=True) for r in read(O/'team-extraction.json')])
 groups=recovery_train.component_groups(rows)
 for r,g in zip(rows,groups):r['group']=g
 assert len({r['id'] for r in rows})==len(rows)
 reserved=read(B/'output/five-exercise-recovery-20261023/final_test_hashed_manifest.json');reserved_sources={x for r in reserved['records'] for x in r.get('sources',[])};assert not reserved_sources&{s for r in rows for s in r['sources']}
 save('score-inputs.frozen.json',rows);save('score-intake-audit.json',dict(rows=len(rows),source_families=dict(collections.Counter(r['family'] for r in rows)),roles=dict(collections.Counter(r['split'] for r in rows)),team_excluded=excluded,person_source_components_no_overlap=True,final_test_pixels_or_labels_opened=False,old_validation_roles_preserved=True,team_validation_person='P04',limits=['Existing and R2D2 labels are nonexpert drafts.','YTW upright features stay separate from legacy hinged criteria.','Front knee-gap is not the legacy combined knee/heel criterion.','Standing return grade 0 has no confirmed examples.','Side-only squat return is not relabeled as frontal height return.']))
 status('score_inputs_ready',rows=len(rows),team=len(team),prior_return=len(prior))
if __name__=='__main__':main()
