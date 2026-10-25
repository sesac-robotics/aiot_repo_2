import json,sys,hashlib,collections
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];O=B/'output/mild-return-mining-v2-20261024/qevd';O.mkdir(exist_ok=True);sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import assert_training_allowed
read=lambda p:json.loads(p.read_text());rows=read(B/'output/mild-return-calibrated-20261024/score-inputs.frozen.json');test=read(B/'output/mild-return-calibrated-20261024/test-inputs.frozen.json');known={s for r in rows+test for s in r['sources']};roles={r['person']:r['split'] for r in rows if r['family']!='R2D2'};protected=set(read(B/'output/five-exercise-recovery-20261023/final_test_reservation.json')['reserved_people'])|{'qevd_'+str(i) for i in read(B/'data/qevd-selected/manifests/official_test_workers.json')}|{'qevd_2785','qevd_956'}
for p in [B/'output/return-root-cause-20261024/mining/intake.frozen.json',B/'output/squat-side-return-v3-20261024/mining/intake.frozen.json']:
 known|={r['source_sha256'] for r in read(p)}
known|={r['source_sha256'] for r in read(B/'output/return-failure-expansion-20261024/intake.frozen.json')['records']}
pool=read(B/'output/squat-errors-20261023/pose-search.json');chosen=[];people=set()
for r in sorted(pool,key=lambda r:(abs((r.get('return_signal') or 0)-.25),-r['frame_count'])):
 if len(chosen)==12:break
 if r['person'] in people or r['person'] in protected or roles.get(r['person'],'train')!='train' or r.get('torso_coverage',0)<.9 or r['frame_count']<155:continue
 digest=sha(r['path'])
 if digest in known:continue
 role=roles.get(r['person'])
 if role is None:role='validation' if int(hashlib.sha256(('side-return-mining-'+r['person']).encode()).hexdigest()[:8],16)%5==0 else 'train'
 if role!='train':continue
 a=dict(r,split=role,source_sha256=digest,review_id=f'Q{len(chosen)+1:02}',selection='unused Train source with small end-height signal, not a failure label');assert_training_allowed([a]);chosen.append(a);known.add(digest);people.add(r['person'])
assert not (O/'intake.frozen.json').exists();(O/'intake.frozen.json').write_text(json.dumps(chosen,indent=2));print('FROZEN',len(chosen),flush=True)
for r in chosen:
 c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);N=int(c.get(7));frames=list(range(0,N,max(1,round(r['fps']/5))));current=-1
 for start in range(0,len(frames),32):
  canvas=Image.new('RGB',(1680,1000),'white');d=ImageDraw.Draw(canvas)
  for j,f in enumerate(frames[start:start+32]):
   while current<f:ok=c.grab();current+=1
   if not ok:break
   ok,a=c.retrieve();assert ok;im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((210,220));x=j%8*210;y=j//8*250;canvas.paste(im,(x+(210-im.width)//2,y+25));d.text((x,y),f"{r['review_id']} f{f} {f/r['fps']:.2f}s",fill='black')
  canvas.save(O/f"{r['review_id']}-{start//32+1:02}.jpg",quality=95)
 c.release();print('PREVIEW',r['review_id'],r['person'],r['frame_count'],flush=True)
