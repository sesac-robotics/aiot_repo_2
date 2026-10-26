import json,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];O=B/'output/mild-return-mining-v2-20261024';p=B/'output/squat-return-20261023/cfrep/squat_front_m4_6_6.mp4';frames=[41]+list(range(268,442,6));c=cv2.VideoCapture(str(p),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);canvas=Image.new('RGB',(1680,1200),'white');d=ImageDraw.Draw(canvas)
for j,f in enumerate(frames):
 c.set(1,f);ok,a=c.read();assert ok;cols=np.flatnonzero((a.max(2)>20).mean(0)>.1)
 if len(cols):a=a[:,cols[0]:cols[-1]+1]
 im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((210,260));x=j%8*210;y=j//8*300;canvas.paste(im,(x+(210-im.width)//2,y+25));d.text((x,y),f'f{f} {f/30:.2f}s',fill='black')
c.release();canvas.save(O/'m4-unassigned.jpg',quality=96)
