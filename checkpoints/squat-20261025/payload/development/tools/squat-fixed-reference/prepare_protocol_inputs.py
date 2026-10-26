import json,sys
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
import shoulder_protocol_features as features
B=Path(__file__).resolve().parents[2];sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
O=B/'output/shoulder-common-support-audit-20261024/full-review-v1/protocol-model-v1'
def main():
 labels=json.loads((O/'labels.frozen.json').read_text())['labels'];rows=[]
 for r in labels:
  p=O/'observations'/r['id']/'side.npz';meta=json.loads(p.with_suffix('.json').read_text());z=np.load(p)
  assert meta['config']['source_sha256']==r['source_sha256'] and str(z['fingerprint'])==meta['fingerprint']
  c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);w=int(c.get(3));h=int(c.get(4));c.release()
  f=features.extract(z['image_xy'],w,h)
  rows.append(dict(r,feature=f,cache=str(p),cache_sha256=sha(p),original_width=w,original_height=h))
  if f['eligible']:
   # Audit the target person/chain against original source pixels before fitting.
   cap=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);out=Image.new('RGB',(960,360),'white');draw=ImageDraw.Draw(out)
   for j,idx in enumerate(np.linspace(0,len(z['frame_numbers'])-1,3).round().astype(int)):
    frame=int(z['frame_numbers'][idx]);cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,a=cap.read();assert ok
    im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((320,320));px=j*320+(320-im.width)//2;py=30;out.paste(im,(px,py));d=ImageDraw.Draw(out)
    factor=im.width/w
    for joint in f['required_joints']:
     v=z['image_xy'][idx,joint]
     if np.isfinite(v).all():
      x,y=px+v[0]*factor,py+v[1]*factor;d.ellipse((x-3,y-3,x+3,y+3),fill='red');d.text((x+3,y),str(joint),fill='red')
    draw.text((j*320+3,5),f"{r['id'][8:11]} {r['protocol_label']} f{frame}",fill='black')
   cap.release();out.save(O/(r['id']+'-tracking.jpg'),quality=96)
 p=O/'inputs.frozen.json';assert not p.exists();p.write_text(json.dumps(dict(feature_names=features.NAMES,feature_version=features.VERSION,
     label_sha256=sha(O/'labels.frozen.json'),rows=rows),ensure_ascii=False,indent=2,allow_nan=False))
 print('INPUTS',len(rows),'eligible',sum(r['feature']['eligible'] for r in rows),'held',[(r['id'],r['feature'].get('reason')) for r in rows if not r['feature']['eligible']],flush=True)
if __name__=='__main__':main()
