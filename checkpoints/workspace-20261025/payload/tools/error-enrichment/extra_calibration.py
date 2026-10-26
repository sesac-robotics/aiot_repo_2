from pathlib import Path
import json,cv2,numpy as np,hashlib
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/error-enrichment-20261024';D=B/'output/squat-return-20261023/cfrep';ids=['squat_front_m1_9_4','squat_front_m4_6_6','squat_front_w1_9_3','squat_front_w2_9_3'];rows=[];canvas=Image.new('RGB',(8*180,4*300),'#dddddd')
for i,id in enumerate(ids):
 p=D/(id+'.mp4');c=cv2.VideoCapture(str(p),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);fps=c.get(5);fs=[0,5,10,15,20,30,45,60]
 for j,f in enumerate(fs):
  c.set(1,f);ok,a=c.read();assert ok
  columns=np.where((a.max(2)>20).mean(0)>.1)[0];a=a[:,columns[0]:columns[-1]+1];im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((180,270));x=j*180;y=i*300;canvas.paste(im,(x+(180-im.width)//2,y+25));ImageDraw.Draw(canvas).text((x+2,y+3),id+' f'+str(f),fill='black')
 c.release();rows.append(dict(id=id,path=str(p),person='cfrep_'+id.split('_')[2],fps=fps,split='validation' if '_w2_' in id else 'train',sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),sheet=str(O/'extra-calibration-review.jpg')))
canvas.save(O/'extra-calibration-review.jpg',quality=92);(O/'extra-calibration-candidates.json').write_text(json.dumps(rows,indent=2))
