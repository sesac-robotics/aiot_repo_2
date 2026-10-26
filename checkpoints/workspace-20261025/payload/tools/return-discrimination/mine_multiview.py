import json,sys,collections
from pathlib import Path
import cv2
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];O=B/'output/mild-return-mining-v2-20261024/multiview';O.mkdir(exist_ok=True);sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import assert_training_allowed
read=lambda p:json.loads(p.read_text());rows=read(B/'output/mild-return-calibrated-20261024/score-inputs.frozen.json');test=read(B/'output/mild-return-calibrated-20261024/test-inputs.frozen.json');known={s for r in rows+test for s in r['sources']};roles={r['person']:r['split'] for r in rows if r['family']=='MultiView'};groups={r['person']:r['group'] for r in rows if r['family']=='MultiView'};protected=set(read(B/'output/five-exercise-recovery-20261023/final_test_reservation.json')['reserved_people']);root=B/'data/A Multi-View Raw Video Dataset of Seven Fitness Ex/Dataset Exercise Quality-wise/squat/bad/side';chosen=[]
for p in sorted(root.glob('*.mp4')):
 person='shoulder_'+p.name.split('_')[1]
 if roles.get(person)!='train' or person in protected:continue
 digest=sha(p)
 if digest in known:continue
 c=cv2.VideoCapture(str(p));N=int(c.get(7));fps=float(c.get(5));c.release();r=dict(id=p.stem,path=str(p),person=person,group=groups[person],split='train',source_sha256=digest,frame_count=N,fps=fps,review_id=f'M{len(chosen)+1:02}',provider_bad_not_a_return_label=True);assert_training_allowed([r]);chosen.append(r);known.add(digest)
 if len(chosen)==4:break
assert not (O/'intake.frozen.json').exists();(O/'intake.frozen.json').write_text(json.dumps(chosen,indent=2));print('FROZEN',len(chosen),flush=True)
for r in chosen:
 c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);frames=list(range(0,r['frame_count']-2,max(1,round(r['fps']/3))));current=-1
 for start in range(0,len(frames),32):
  canvas=Image.new('RGB',(1680,1000),'white');d=ImageDraw.Draw(canvas)
  for j,f in enumerate(frames[start:start+32]):
   while current<f:ok=c.grab();current+=1
   if not ok:break
   ok,a=c.retrieve();assert ok;im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((210,220));x=j%8*210;y=j//8*250;canvas.paste(im,(x+(210-im.width)//2,y+25));d.text((x,y),f"{r['review_id']} f{f} {f/r['fps']:.2f}s",fill='black')
  canvas.save(O/f"{r['review_id']}-{start//32+1:02}.jpg",quality=95)
 c.release();print('PREVIEW',r['review_id'],r['person'],len(frames),flush=True)
