from pathlib import Path
import cv2,json
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent
p=R/'data/A Multi-View Raw Video Dataset of Seven Fitness Ex/Dataset Exercise Quality-wise/squat/bad/side/subject_014_squat_bad_side.mp4'
c=cv2.VideoCapture(str(p));fps=c.get(cv2.CAP_PROP_FPS);frames=list(range(0,181,6));canvas=Image.new('RGB',(8*180,4*270),'#eef3f5');d=ImageDraw.Draw(canvas)
for j,n in enumerate(frames):
 c.set(cv2.CAP_PROP_POS_FRAMES,n);ok,bgr=c.read()
 if not ok:continue
 im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));im.thumbnail((176,240));x=(j%8)*180;y=(j//8)*270;canvas.paste(im,(x,y+22));d.text((x+4,y+3),f'f{n} / {n/fps:.2f}s',fill='black')
canvas.save(O/'return014-review.jpg');c.release();print(fps)
