import json,cv2,numpy as np
from PIL import Image,ImageDraw
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/pushup-root-cause-20261024';atlas=json.loads((O/'source-atlas.json').read_text());plans={r['id']:r for r in json.loads((O/'crop-plan-v2.frozen.json').read_text())}
for index in [16,38]:
 r=plans[atlas[index-1]['id']];c=cv2.VideoCapture(r['source'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);s=r['segment'];fs=np.linspace(s['start_frame'],s['end_frame'],12).round().astype(int);canvas=Image.new('RGB',(1800,4*290),'white')
 for k,f in enumerate(fs):
  c.set(1,int(f));ok,a=c.read();assert ok;im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((600,260));x=k%3*600;y=k//3*290;canvas.paste(im,(x,y+25));ImageDraw.Draw(canvas).text((x+5,y+5),f'D{index:03} {f/s["fps"]:.3f}s',fill='black')
 c.release();canvas.save(O/f'D{index:03}-detail.jpg',quality=94)
