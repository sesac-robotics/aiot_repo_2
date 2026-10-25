import json,cv2
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];O=B/'output/mild-return-mining-v2-20261024/qevd-more';rs=json.loads((O/'intake.frozen.json').read_text())
for r in rs:
 if r['review_id'] not in ['Q09','Q11']:continue
 frames=[0,6]+list(range(60,85,3))+list(range(120,145,3));canvas=Image.new('RGB',(5*384,4*270),'white');d=ImageDraw.Draw(canvas);c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
 for j,f in enumerate(frames):
  c.set(1,f);ok,a=c.read();assert ok;im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((384,240));x=j%5*384;y=j//5*270;canvas.paste(im,(x+(384-im.width)//2,y+22));d.text((x,y),f'{r["review_id"]} f{f}',fill='black')
 c.release();canvas.save(O/f'{r["review_id"]}-dense.jpg',quality=96)
