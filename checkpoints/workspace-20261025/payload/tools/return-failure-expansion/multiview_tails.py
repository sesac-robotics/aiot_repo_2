"""Inspect unreviewed later repetitions; preserve existing MultiView person roles."""
import json,sys,cv2,numpy as np,os
from pathlib import Path
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/return-failure-expansion-20261024/tail-review';O.mkdir(exist_ok=True);sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
pool=json.loads((B/'output/overnight-20261020/score_candidates.json').read_text());roles={r['id'].split('_')[1]:r['split'] for r in pool if r['exercise']=='shoulder'};root=B/'data/A Multi-View Raw Video Dataset of Seven Fitness Ex/Dataset Exercise-wise';selected=[]
for folder,exercise,view in [('push_up','pushups','side'),('shoulder','shoulder','front')]:
 for num,role in sorted(roles.items()):
  if role=='test':continue
  sid=f'subject_{num}_{folder}_bad_{view}';path=root/folder/'bad'/view/(sid+'.mp4')
  if not path.exists():continue
  cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);fps=cap.get(5);count=int(cap.get(7));cap.release();duration=count/fps;start=max(0,duration-14);end=duration-.15;selected.append(dict(id=sid,review_id=f'T{len(selected)+1:03}',person='shoulder_'+num,split=role,exercise=exercise,view=view,path=str(path),source_sha256=sha(path),fps=fps,frame_count=count,duration=duration,start=start,end=end,selection_reason='Train_support_deficit_later_raw_repetitions_not_Test_error_or_prediction',selection_not_a_label=True))
p=O/'intake.frozen.json'
if p.exists():assert json.loads(p.read_text())==selected
else:p.write_text(json.dumps(selected,indent=2))
for k,r in enumerate(selected):
 if k<int(os.environ.get('TAIL_REVIEW_RESUME','0')):continue
 if k%4==0:canvas=Image.new('RGB',(12*160,4*245),'white')
 draw=ImageDraw.Draw(canvas);row=k%4;y=row*245;draw.text((3,y+2),r['review_id']+' '+r['id']+' '+r['split'],fill='black');cap=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);frames=np.linspace(round(r['start']*r['fps']),round(r['end']*r['fps']),12).round().astype(int)
 cap.set(1,int(frames[0]));current=int(frames[0])-1
 for j,f in enumerate(frames):
  while current<f:ok=cap.grab();current+=1;assert ok
  ok,a=cap.retrieve();assert ok;im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((160,195));canvas.paste(im,(j*160+(160-im.width)//2,y+23));draw.text((j*160+2,y+221),f'{f/r["fps"]:.2f}s',fill='black')
 cap.release();canvas.save(O/f'tails-{k//4+1:02}.jpg',quality=93);print('TAIL',r['review_id'],r['id'],flush=True)
print('TAIL_INTAKE_COMPLETE',len(selected),flush=True)
