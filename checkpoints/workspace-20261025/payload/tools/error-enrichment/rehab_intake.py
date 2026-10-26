from pathlib import Path
import json,cv2,hashlib,numpy as np
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/error-enrichment-20261024';r=json.loads((B/'mediapipe-pilot/correction/sample-manifest.json').read_text());selected=[]
for person in ['Petra','Terka','Vasil','Mima']:
 for label,n in [('NORMAL',2),('ERR_KNEES_INSIDE',1),('ERR_TORSO_TO_SIDE',1)]:
  for s in [a for a in r if a['person']==person and a['label']==label and a['orientation']=='0' and a['camera']=='1'][:n]:
   s=dict(s);s.update(review_id='R'+str(len(selected)+1).zfill(3),path=str(B/'data/rehab26-squat/raw/videos/ex6'/(s['video_id']+'.mp4')));s['person']='rehab_'+person;selected.append(s)
for i,s in enumerate(selected):
 c=cv2.VideoCapture(s['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);fps=c.get(5);s['fps']=fps;s['source_sha256']=hashlib.file_digest(Path(s['path']).open('rb'),'sha256').hexdigest();fs=np.linspace(s['first_frame'],s['last_frame'],8).round().astype(int);s['review_frames']=fs.tolist()
 if i%4==0:canvas=Image.new('RGB',(1600,4*290),'#dddddd')
 for j,f in enumerate(fs):
  c.set(1,int(f));ok,a=c.read();assert ok
  im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((200,265));x=j*200;y=(i%4)*290;canvas.paste(im,(x+(200-im.width)//2,y+23));ImageDraw.Draw(canvas).text((x+2,y+3),s['review_id']+' f'+str(f),fill='black')
 c.release();s['sheet']=str(O/f'rehab-review-{i//4+1:02}.jpg');canvas.save(s['sheet'],quality=92)
(O/'rehab-intake.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2));print('REHAB26 prepared',len(selected),'official test not opened')
