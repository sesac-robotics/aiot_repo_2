"""Append approved Train-only return cycles; keep the frozen Test byte-identical."""
import json,sys,copy,shutil,datetime
from pathlib import Path
import numpy as np,cv2
B=Path('/home/jayden/SeSACPAI-Mid');D=B/'output/return-failure-expansion-20261024';OLD=B/'output/paired-failure-enriched-five-v2-20261024';O=B/'output/return-failure-scores-20261024';O.mkdir(exist_ok=True);SUB=O/'changed-heads-fit';SUB.mkdir(exist_ok=True)
sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'tools/integrated-five'),str(B/'five-exercise-comparison')]
from common import sha
from split_guard import assert_training_allowed,POLICY
from team_features import describe
import three_items
read=lambda p:json.loads(p.read_text());rows=read(OLD/'score-inputs.frozen.json');groups={r['person']:r['group'] for r in rows};new=[];audit=[];changed=set()
for d in read(D/'labels-v2.frozen.json')['reviewed']:
 assert d['train_eligible'] and d['whole_attempt_observed'];assert sha(d['path'])==d['source_sha256'];p=D/'observations'/d['review_id']/'primary.npz';meta=read(p.with_suffix('.json'));z=np.load(p);assert meta['config']['source_sha256']==d['source_sha256'] and str(z['fingerprint'])==meta['fingerprint'];ts=z['frame_numbers']/d['fps'];seq=z['image_xy'][(ts>=d['start'])&(ts<d['end'])].copy();cap=cv2.VideoCapture(d['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);w,h=cap.get(3),cap.get(4);cap.release();seq=three_items.mask_image_border(seq,w,h);notes=describe('pushups',seq,'front',d['end']-d['start'],None,w,h)
 for n,g in zip(notes,d['grades']):
  for k in ['rule_grade','measurement','higher_is_better','proxy_only']:n.pop(k,None)
  if g is None:n.update(eligible=False,feature=None,holds=['criterion_not_labeled_from_visible_video'])
  if n['feature'] is not None and not np.isfinite(n['feature']).all():n.update(eligible=False,feature=None,holds=['nonfinite_required_pose'])
 r=dict(id=d['id'],exercise='pushups',person=d['person'],group=groups.get(d['person'],d['person']),sources=[d['source_sha256']],recording_groups=[d['person']],split='train',levels=d['grades'],items=notes,family='QEVD',origin='return_failure_expansion',view='front',variant=d['variant'],label_origin='8fps_direct_visual_draft',reference_total_allowed=False,whole_attempt_observed=True,source_annotation=d['review_id']);assert_training_allowed([r]);rows.append(r);new.append(r);changed|={(r['exercise'],n['item'],n['bank']) for n,g in zip(notes,r['levels']) if n['eligible'] and g is not None};audit.append(dict(id=r['id'],review_id=d['review_id'],cache_sha256=sha(p),start=d['start'],end=d['end'],return_grade=d['grades'][2],return_feature_observed=notes[2]['eligible'],terminal_tracking_sheet_sha256=sha(D/f"tracking-v2-{(len(new)-1)//3+1:02}.jpg")))
assert len({r['id'] for r in rows})==len(rows);assert changed=={('pushups','return_extension','front')};assert_training_allowed(new)
for n,v in [('score-inputs.frozen.json',rows),('feature-provenance.json',audit)]:
 p=O/n;assert not p.exists();p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
for n in ['test-inputs.frozen.json','split-audit.json']:shutil.copyfile(OLD/n,O/n)
subset=[]
for r in rows:
 s=copy.deepcopy(r);pairs=[(n,g) for n,g in zip(s['items'],s['levels']) if (r['exercise'],n['item'],n['bank']) in changed]
 if pairs:s['items']=[n for n,g in pairs];s['levels']=[g for n,g in pairs];subset.append(s)
(SUB/'score-inputs.frozen.json').write_text(json.dumps(subset,ensure_ascii=False,indent=2,allow_nan=False));(O/'MODEL_SELECTION_PROTOCOL.frozen.json').write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_run=str(OLD),changed_keys=sorted(changed),train_input_sha256=sha(O/'score-inputs.frozen.json'),test_input_sha256=sha(O/'test-inputs.frozen.json'),policy_sha256=sha(POLICY),labels_sha256=sha(D/'labels-v2.frozen.json'),selection='unchanged Train group CV and fixed development gates; no Test-driven candidate or threshold selection',production_replacement=False),indent=2));print('PREPARED',len(rows),'NEW',len(new),'observable',sum(r['items'][2]['eligible'] for r in new),flush=True)
