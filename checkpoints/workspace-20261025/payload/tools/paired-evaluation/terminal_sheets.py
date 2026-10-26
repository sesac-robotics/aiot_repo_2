import json,cv2,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/ai-rule-paired-20261024';rows=[r for r in json.loads((O/'reviewed-intake-v3.frozen.json').read_text())['labels'] if r['train_eligible']]
for k,r in enumerate(rows):
 cap=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
 if k%4==0:sheet=Image.new('RGB',(1200,1000),'white')
 # Descriptor uses at most five last 10Hz samples (0.5s), or last15% for short clips.
 tail=min(.5,max(.2,round((r['end']-r['start'])*10*.15)/10))
 for j,t in enumerate(np.linspace(r['end']-tail,r['end']-.05,4)):
  cap.set(1,round(t*r['fps']));ok,a=cap.read();assert ok;im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((295,220));x=j*300;y=k%4*250;sheet.paste(im,(x,y+25));ImageDraw.Draw(sheet).text((x+4,y+3),f"{r['review_id']} {t:.2f}s",fill='black')
 cap.release();sheet.save(O/f'terminal-v3-{k//4+1:02}.jpg',quality=94)
print('TERMINAL_SHEETS',4)
