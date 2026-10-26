"""Preassign source roles, then show new existing squat originals for review."""
import json,sys,hashlib
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];O=B/'output/return-root-cause-20261024/mining';O.mkdir(exist_ok=True)
sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
read=lambda p:json.loads(p.read_text())
old=read(B/'output/squat-side-return-enriched-v2-20261024/score-inputs.frozen.json');test=read(B/'output/return-failure-scores-20261024/test-inputs.frozen.json');known={s for r in old+test for s in r['sources']};roles={r['person']:r['split'] for r in old if r['family']!='R2D2'}
protected=set(read(B/'output/five-exercise-recovery-20261023/final_test_reservation.json')['reserved_people'])|{'qevd_'+str(i) for i in read(B/'data/qevd-selected/manifests/official_test_workers.json')}|{'qevd_2785','qevd_956'}
for path in [B/'output/return-failure-expansion-20261024/intake.frozen.json']:
    known|={r['source_sha256'] for r in read(path)['records']}
known|={r['source_sha256'] for r in read(B/'output/squat-side-return-v3-20261024/mining/intake.frozen.json')}
pool=read(B/'output/squat-errors-20261023/pose-search.json');chosen=[];people=set()
for r in sorted(pool,key=lambda r:-(r.get('return_signal') or 0)):
    if len(chosen)==8:break
    p=r['person']
    if p in people or p in protected or roles.get(p,'train')!='train' or r.get('torso_coverage',0)<.8:continue
    digest=sha(r['path'])
    if digest in known:continue
    role=roles.get(p)
    if role is None:role='validation' if int(hashlib.sha256(('side-return-mining-'+p).encode()).hexdigest()[:8],16)%5==0 else 'train'
    chosen.append(dict(r,review_id=f'R{len(chosen)+1:02}',split=role,source_sha256=digest,selection='remaining_Train_pool_height_hint_only_not_a_label',role_frozen_before_visual_review=True));people.add(p);known.add(digest)
assert len(chosen)==8 and not (O/'intake.frozen.json').exists()
(O/'intake.frozen.json').write_text(json.dumps(chosen,indent=2))
for r in chosen:
    cap=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);frames=list(range(0,r['frame_count'],max(1,round(r['fps']/4))));current=-1
    for offset in range(0,len(frames),32):
        canvas=Image.new('RGB',(8*210,4*225),'white');d=ImageDraw.Draw(canvas)
        for j,f in enumerate(frames[offset:offset+32]):
            while current<f:ok=cap.grab();current+=1;assert ok
            ok,a=cap.retrieve();assert ok;im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((210,198));x=j%8*210;y=j//8*225;canvas.paste(im,(x+(210-im.width)//2,y+24));d.text((x,y),f"{r['review_id']} {f/r['fps']:.2f}s f{f}",fill='black')
        canvas.save(O/f"{r['review_id']}-{offset//32+1:02}.jpg",quality=95)
    cap.release();print(r['review_id'],r['id'],r['person'],r['split'],len(frames),flush=True)
