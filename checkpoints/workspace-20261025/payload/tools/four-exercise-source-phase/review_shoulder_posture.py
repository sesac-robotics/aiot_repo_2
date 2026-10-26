from pathlib import Path
import json,cv2
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/four-exercise-source-phase-v2-20261025';rows=json.loads((O/'all-resolved-inputs.frozen.json').read_text());unique={}
for r in rows:
 if r['exercise']=='shoulder':unique.setdefault(r['source_sha256'],r)
plan=sorted(unique.values(),key=lambda r:(r['person'],r['source_sha256']));receipts=[]
for page in range((len(plan)+5)//6):
 canvas=Image.new('RGB',(1200,1000),'white');draw=ImageDraw.Draw(canvas)
 for k,r in enumerate(plan[page*6:page*6+6]):
  f=r['actual_native_frames'][len(r['actual_native_frames'])//2];c=cv2.VideoCapture(r['source_path']);c.set(1,f);ok,fr=c.read();c.release();assert ok;im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB));im.thumbnail((390,425));x=k%3*400+(400-im.width)//2;y=k//3*500+60;canvas.paste(im,(x,y));draw.text((k%3*400+3,k//3*500+5),f"#{page*6+k} {r['person']} f{f}\n{Path(r['source_path']).stem[:48]}",fill='black');receipts.append(dict(index=page*6+k,person=r['person'],source_sha256=r['source_sha256'],source_path=r['source_path'],native_frame=f,page=f'review/posture-{page:02d}.jpg',scope='posture_variant_only_not_severity_or_whole_clip_review'))
 canvas.save(O/f'review/posture-{page:02d}.jpg',quality=93)
(O/'shoulder-posture-review-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n');print(len(receipts))
