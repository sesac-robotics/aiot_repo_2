from pathlib import Path
import json,cv2,numpy as np
O=Path(__file__).resolve().parent;manifest=[]
for p in sorted((O/'cfrep').glob('*.json')):
 d=json.loads(p.read_text());name=p.name.split('_hrnet')[0];c=cv2.VideoCapture(str(O/'cfrep'/f'{name}.mp4'));fps=c.get(5);N=int(c.get(7));print(name,'fps',fps,'N',N)
 for page in range((len(d['annotations'])+3)//4):
  canvas=np.full((4*240,9*200,3),245,np.uint8)
  for row,ann in enumerate(d['annotations'][page*4:page*4+4]):
   num=page*4+row+1;frames=np.linspace(max(0,ann['start']-4),min(N-1,ann['end']+5),9).round().astype(int);r=dict(id=f'{name}_r{num:02}',person='cfrep_'+name.split('_')[2],split='validation' if '_w2_' in name else 'train',path=str(O/'cfrep'/f'{name}.mp4'),first_frame=ann['start'],last_frame=ann['end'],fps=fps,provider_hint=ann['label'],review_frames=frames.tolist(),sheet=str(O/f'{name}-{page}.jpg'));manifest.append(r)
   for col,k in enumerate(frames):
    c.set(1,int(k));ok,im=c.read()
    if not ok:raise RuntimeError(r['id'])
    cols=np.where((im.max(2)>20).mean(0)>.1)[0];im=im[:,cols[0]:cols[-1]+1] if len(cols) else im;scale=min(200/im.shape[1],210/im.shape[0]);im=cv2.resize(im,(round(im.shape[1]*scale),round(im.shape[0]*scale)));x=col*200;y=row*240;canvas[y+27:y+27+im.shape[0],x:x+im.shape[1]]=im;cv2.putText(canvas,f'r{num:02} f{k} {ann["label"]}',(x+2,y+17),0,.4,(0,0,0),1)
  cv2.imwrite(str(O/f'{name}-{page}.jpg'),canvas)
 c.release()
(O/'cfrep-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
