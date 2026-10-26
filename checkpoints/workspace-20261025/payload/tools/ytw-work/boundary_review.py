"""Pixel-only source review, no model predictions drive sheet times."""
import json,hashlib,math
from pathlib import Path
from fractions import Fraction
from datetime import datetime,timezone
import cv2
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/ytw-work-20261024/boundary-v2';O.mkdir(exist_ok=True)
I={r['id']:r for r in json.load(open(B/'output/r2d2-review-20261024/inventory.json'))};policy=json.load(open(B/'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json'))
# Centers from prior direct pixel review, not classifier predictions.
centers={'V37':[[3.6,4.7,5.9],[8.1,9.2,10.9],[12.8,14.4,15.6]],'V39':[[4.4,6.4,7.6],[9.4,10.9,12.1],[13.4,15.4,16.6]],'V41':[[3.6,5.2,6.1],[7.9,8.9,9.8]],'V43':[[5.8,7.1,8.3],[10.1,11.2,12.1]]}
plan=dict(schema='ytw-video-boundary-review-plan-v2',frozen_at_utc=datetime.now(timezone.utc).isoformat(),scope='fixed_person_excluded_development_exposed_AI_review_not_independent_final',source_selection={'boundaries':list(centers),'negative_screen':['V38','V40','V44'],'Test_excluded':['V42']},centers=centers,entry_definition='first visually unambiguous stage attempt held for >=5 actual 10Hz samples; short transition or incomplete visibility is uncertain, not grade0',phase_quality='stage attempt, not correct exercise morphology or severity',boundary_reference='start=Y entry; phase starts=Y/T/W entry; terminal=W entry+4slots; confirmation latency explicitly separate',uncertainty='store entry bracket; if not unambiguous inside +/-0.4s strip expand pixel review; never infer from model output',metrics='compare fixed candidate to video drafts; exact expert ground truth unavailable; uncertainty bracket and scorer/quality support separate',negative_policy='natural contiguous wait/aborted/wrong-order only; no slicing one wait into many, no bad-quality-label copying to incomplete',tracker_frozen_sha256=hashlib.sha256((B/'output/live-webcam-20261023/ytw_candidate.py').read_bytes()).hexdigest(),models_changed=False)
p=O/'plan.frozen.json'
if not p.exists():p.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
else:assert json.load(open(p))['centers']==centers
manifest=[]
for sid in ['V38','V40','V44']+list(centers):
 r=I[sid];assert policy['assignments'][sid]=='train';assert hashlib.sha256(Path(r['copy']).read_bytes()).hexdigest()==r['sha256']
 fps=float(Fraction(r['fps']));cap=cv2.VideoCapture(r['copy']);requests={}
 if sid in centers:
  for cycle,cs in enumerate(centers[sid],1):
   for phase,c in zip('YTW',cs):
    for tick in range(round(c*10)-4,round(c*10)+5):requests[round(tick/10*fps)]=tick/10
 else:
  for i in range(math.ceil(r['duration']*2)):requests[round(i*.5*fps)]=i*.5
 images={};index=0
 while True:
  ok,a=cap.read()
  if not ok:break
  if index in requests:
   im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((156,185));images[round(requests[index],3)]=(im.copy(),index)
  index+=1
 cap.release()
 if sid in centers:
  for cycle,cs in enumerate(centers[sid],1):
   canvas=Image.new('RGB',(1440,660),'white');d=ImageDraw.Draw(canvas)
   for row,(phase,c) in enumerate(zip('YTW',cs)):
    for col,tick in enumerate(range(round(c*10)-4,round(c*10)+5)):
     t=round(tick/10,3);im,frame=images[t];x=col*160;y=row*220
     canvas.paste(im,(x+(160-im.width)//2,y+30));d.text((x+3,y+3),f'{phase} {t:.1f}s f{frame}',fill='black')
   p=O/f'{sid}-cycle{cycle}-entries.jpg';canvas.save(p,quality=94)
   manifest.append(dict(source_id=sid,source_sha256=r['sha256'],sheet=str(p),kind='10Hz_transition_strip',cycle=cycle,centers=cs))
 else:
  ts=sorted(images)
  for page in range(math.ceil(len(ts)/35)):
   canvas=Image.new('RGB',(1120,1050),'white');d=ImageDraw.Draw(canvas)
   for j,t in enumerate(ts[page*35:(page+1)*35]):
    im,frame=images[t];x=j%7*160;y=j//7*210;canvas.paste(im,(x+(160-im.width)//2,y+25));d.text((x+3,y+3),f'{sid} {t:.1f}s',fill='black')
   p=O/f'{sid}-screen{page+1}.jpg';canvas.save(p,quality=92)
   manifest.append(dict(source_id=sid,source_sha256=r['sha256'],sheet=str(p),kind='0.5s_negative_screen'))
 print('REVIEW_READY',sid,flush=True)
(O/'sheets.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
