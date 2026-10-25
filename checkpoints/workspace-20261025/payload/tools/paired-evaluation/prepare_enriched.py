import json,sys,copy,shutil,datetime
from pathlib import Path
import numpy as np,cv2
B=Path('/home/jayden/SeSACPAI-Mid');D=B/'output/ai-rule-paired-20261024';OLD=B/'output/pushup-repaired-five-v2-20261024';O=B/'output/paired-enriched-five-v2-20261024';O.mkdir(exist_ok=True);SUB=O/'return-only-fit';SUB.mkdir(exist_ok=True)
sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'five-exercise-comparison')]
from common import sha
from split_guard import assert_training_allowed,POLICY
import three_items
rows=json.loads((OLD/'score-inputs.frozen.json').read_text());oldpeople={r['person']:r['group'] for r in rows};review=json.loads((D/'reviewed-intake-v3.frozen.json').read_text())['labels'];provenance=[]
for r in review:
 if not r['train_eligible']:continue
 p=D/'observations-v2'/r['review_id']/'primary.npz';m=json.loads(p.with_suffix('.json').read_text());assert m['config']['source_sha256']==r['source_sha256'];z=np.load(p);assert str(z['fingerprint'])==m['fingerprint'];ts=z['frame_numbers']/r['fps'];seq=z['image_xy'][(ts>=r['start'])&(ts<r['end'])]
 cap=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);width,height=cap.get(3),cap.get(4);cap.release();seq=three_items.mask_image_border(seq,width,height);notes=three_items.describe('pushups',seq,None)
 for n,g in zip(notes,r['levels']):
  n['bank']='front';n.pop('measurement',None);n.pop('higher_is_better',None)
  if g is None:n.update(eligible=False,feature=None,holds=['direct_review_item_unobservable_or_grade_uncertain'])
  if n['feature'] is not None and not np.isfinite(n['feature']).all():n.update(eligible=False,feature=None,holds=['nonfinite_observation'])
 out=dict(id=r['id'],exercise='pushups',person=r['person'],group=oldpeople.get(r['person'],r['person']),sources=[r['source_sha256']],recording_groups=[r['person']],split='train',levels=r['levels'],items=notes,origin='paired_rule_train_enrichment',family='QEVD',view='front',variant=r['variant'],label_origin='direct_visual_draft',reference_total_allowed=False,whole_attempt_observed=True,source_annotation=r['review_id'])
 assert_training_allowed([out]);rows.append(out);provenance.append(dict(id=r['id'],source_sha256=r['source_sha256'],cache_sha256=sha(p),cache_metadata_sha256=sha(p.with_suffix('.json')),eligible_return=notes[2]['eligible'],return_holds=notes[2]['holds']))
# Preserve all original inputs and every Test byte exactly; immutable parent remains intact.
for name,value in [('score-inputs.frozen.json',rows),('new-feature-provenance.json',provenance)]:
 p=O/name;assert not p.exists();p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
shutil.copyfile(OLD/'test-inputs.frozen.json',O/'test-inputs.frozen.json');shutil.copyfile(OLD/'split-audit.json',O/'split-audit.json')
subset=[]
for r in rows:
 s=copy.deepcopy(r);pairs=[(n,g) for n,g in zip(s['items'],s['levels']) if n['item']=='return_extension' and n['bank']=='front'];s['items']=[n for n,g in pairs];s['levels']=[g for n,g in pairs]
 if pairs:subset.append(s)
(SUB/'score-inputs.frozen.json').write_text(json.dumps(subset,ensure_ascii=False,indent=2,allow_nan=False))
protocol=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_run=str(OLD),train_input_sha256=sha(O/'score-inputs.frozen.json'),test_input_sha256=sha(O/'test-inputs.frozen.json'),policy_sha256=sha(POLICY),manual_review_sha256=sha(D/'reviewed-intake-v3.frozen.json'),changed_head=['pushups','return_extension','front'],unchanged_heads_copied_from_parent=True,selection='same_Train_group_CV_and_fixed_development_gates',test_selection=False,changes_chosen_from_direct_Train_review=True,all_five_reliable=False,production_replacement=False)
(O/'MODEL_SELECTION_PROTOCOL.frozen.json').write_text(json.dumps(protocol,indent=2));print('PREPARED',len(rows),'new',len(provenance),'eligible',sum(p['eligible_return'] for p in provenance),flush=True)
