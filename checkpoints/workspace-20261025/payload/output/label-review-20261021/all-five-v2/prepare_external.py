"""공개 출처 플랭크 영상에서 시간순 구도 검토판을 추출한다. 점수는 생성하지 않는다."""
from pathlib import Path
import sys
import cv2
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
filename=sys.argv[1] if len(sys.argv)>1 else 'navy_plank.mp4'
prefix=Path(filename).stem
step=float(sys.argv[2]) if len(sys.argv)>2 else 1.0
cap=cv2.VideoCapture(str(ROOT/'sources'/filename),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);fps=cap.get(5);total=int(cap.get(7));ims=[]
if not cap.isOpened() or fps<=0 or total<=0:raise RuntimeError(filename)
for i in range(total):
    ok,im=cap.read()
    if not ok:break
    if i%round(fps*step)!=0:continue
    im=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));im.thumbnail((200,250));ims.append((i,im))
cap.release()
for start in range(0,len(ims),24):
    canvas=Image.new('RGB',(1200,4*285),'#14212a');draw=ImageDraw.Draw(canvas)
    for j,(i,im) in enumerate(ims[start:start+24]):
        x=j%6*200;y=j//6*285;canvas.paste(im,(x,y));draw.text((x,y+252),f'f{i} {i/fps:.2f}s',font=FONT,fill='white')
    canvas.save(ROOT/f'{prefix}_overview_{start//24+1}.jpg',quality=94)
print(fps,total)
