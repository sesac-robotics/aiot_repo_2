import json,copy,sys,datetime
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];D=B/'output/return-label-expansion-v3-20261024';P=B/'output/return-initial-context-20261024';O=B/'output/return-label-enriched-v3-20261024';O.mkdir(exist_ok=True);S=O/'changed-heads-fit';S.mkdir(exist_ok=True)
sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'tools/integrated-five'),str(B/'five-exercise-comparison')]
from common import sha
from split_guard import POLICY,assert_training_allowed
from team_features import describe
import three_items
read=lambda p:json.loads(p.read_text());rows=read(P/'score-inputs.frozen.json');groups={r['person']:r['group'] for r in rows};labels=read(D/'labels.frozen.json')['reviewed'];new=[];audit=[]
for d in labels:
 p=D/'observations'/d['review_id']/'primary.npz';z=np.load(p);meta=read(p.with_suffix('.json'));assert meta['config']['source_sha256']==sha(d['path'])==d['source_sha256'];assert str(z['fingerprint'])==meta['fingerprint']
 frames=z['frame_numbers'];mask=(frames/d['fps']>=d['start'])&(frames/d['fps']<d['end']);seq=z['image_xy'][mask].copy();cap=cv2.VideoCapture(d['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);w,h=cap.get(3),cap.get(4);seq=three_items.mask_image_border(seq,w,h);notes=describe('pushups',seq,'front',d['end']-d['start'],None,w,h)
 for i,g in zip(notes,d['grades']):
  for k in ['rule_grade','measurement','higher_is_better','proxy_only']:i.pop(k,None)
  if g is None:i.update(eligible=False,feature=None,holds=['criterion_not_labeled_from_visible_video'])
  if i['feature'] is not None and not np.isfinite(i['feature']).all():i.update(eligible=False,feature=None,holds=['nonfinite_required_pose'])
 row=dict(id=d['id'],exercise='pushups',person=d['person'],group=groups.get(d['person'],d['person']),sources=[d['source_sha256']],recording_groups=[d['person']],split='train',levels=d['grades'],items=notes,family='QEVD',origin='return_label_expansion_v3',view='front',variant=d['variant'],label_origin=d['annotation_status'],source_annotation=d['review_id'],whole_attempt_observed=True,reference_total_allowed=False)
 assert_training_allowed([row]);new.append(row);audit.append(dict(id=row['id'],person=row['person'],split=row['split'],source_sha256=d['source_sha256'],cache_sha256=sha(p),start=d['start'],end=d['end'],grade=d['return_grade'],eligible=notes[2]['eligible'],feature=notes[2]['feature']))
 canvas=Image.new('RGB',(1200,320),'white');draw=ImageDraw.Draw(canvas)
 for j,f in enumerate(d['review_top_frames'][:3]):
  ix=int(np.argmin(abs(frames-f)));actual=int(frames[ix]);cap.set(1,actual);ok,a=cap.read();assert ok;xy=z['image_xy'][ix]
  for chain in [[11,13,15],[12,14,16]]:
   for v,u in zip(chain,chain[1:]):
    if np.isfinite(xy[[v,u]]).all():cv2.line(a,tuple(xy[v].astype(int)),tuple(xy[u].astype(int)),(0,255,0),2)
  im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((400,280));canvas.paste(im,(j*400,30));draw.text((j*400,2),f"{d['id']} f{actual} grade{d['return_grade']}",fill='black')
 cap.release();canvas.save(D/(d['id']+'-tracking.jpg'),quality=95)
rows+=new;assert len({r['id'] for r in rows})==len(rows);assert_training_allowed([r for r in rows if r['split']=='train'])
changed={('pushups','return_extension','front')};subset=[]
for r in rows:
 pairs=[(i,g) for i,g in zip(r['items'],r['levels']) if (r['exercise'],i['item'],i['bank']) in changed]
 if pairs:
  a=copy.deepcopy(r);a['items']=[i for i,g in pairs];a['levels']=[g for i,g in pairs];subset.append(a)
for p,v in [(O/'score-inputs.frozen.json',rows),(O/'feature-provenance.json',audit),(S/'score-inputs.frozen.json',subset)]:
 assert not p.exists();p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
(O/'test-inputs.frozen.json').write_bytes((P/'test-inputs.frozen.json').read_bytes())
(O/'MODEL_SELECTION_PROTOCOL.frozen.json').write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_run=str(P),policy_sha256=sha(POLICY),reviewed_labels_sha256=sha(D/'labels.frozen.json'),changed_keys=[list(k) for k in changed],new_interval_labels=4,new_partial_failure_intervals=2,new_error_people=1,Train_only_additions=True,Test_byte_identical=True,selection='unchanged generic Train group-CV candidate grid/criterion and fixed development gate; no Test tuning',production_replacement=False),indent=2))
print('PREPARED',len(rows),'added4',audit,flush=True)
