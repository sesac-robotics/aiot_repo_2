import json,cv2,numpy as np
from PIL import Image,ImageDraw
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/pushup-root-cause-20261024';rows=[r for r in json.loads((O/'crop-plan.frozen.json').read_text()) if r['split']!='test'];manifest=[]
for k,r in enumerate(rows):
 if k%6==0:canvas=Image.new('RGB',(1500,6*210),'white')
 c=cv2.VideoCapture(r['source'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);seg=r['segment'];fs=np.linspace(seg['start_frame'],seg['end_frame'],6).round().astype(int);x0,y0,x1,y1=r['crop'];samples=[]
 for j,f in enumerate(fs):
  c.set(1,int(f));ok,a=c.read();assert ok;tile=Image.fromarray(cv2.cvtColor(a[y0:y1,x0:x1],cv2.COLOR_BGR2RGB));tile.thumbnail((250,185));canvas.paste(tile,(j*250,(k%6)*210+25));ImageDraw.Draw(canvas).text((j*250+3,(k%6)*210+4),f'D{k+1:03} {f/seg["fps"]:.2f}s',fill='black');samples.append(int(f))
 c.release();sheet=O/f'source-atlas-{k//6+1:02}.jpg';canvas.save(sheet,quality=91);manifest.append(dict(review_id=f'D{k+1:03}',id=r['id'],split=r['split'],sheet=str(sheet),row=k%6,sampled_frames=samples,previous_labels_hidden=True))
(O/'source-atlas.json').write_text(json.dumps(manifest,indent=2));print('ATLAS',len(rows),flush=True)
