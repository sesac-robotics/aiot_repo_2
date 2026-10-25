from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-single-audit-v1-20261025';rows=json.loads((O/'train-provenance-audit.json').read_text())['rows'];selected=[r for r in rows if r.get('cache') and r['id']==r['source_id'] and r['view']=='front'];receipts=[]
for start in range(0,len(selected),3):
 group=selected[start:start+3];can=Image.new('RGB',(1600,430*len(group)),'#ffffff');dr=ImageDraw.Draw(can)
 for rownum,r in enumerate(group):
  z=np.load(r['cache']);fs=z['frame_numbers'];xy=z['image_xy'];idx=np.unique(np.r_[np.linspace(0,len(fs)-1,5).round().astype(int),np.arange(max(0,len(fs)-3),len(fs))]);cap=cv2.VideoCapture(r['source_path']);nativefps=cap.get(5)
  dr.text((4,rownum*430+4),r['id']+' | original normal labels | red=pose | last 3 samples at right',fill='black')
  for j,i in enumerate(idx):
   native=int(round(float(fs[i])/r['source_fps']*nativefps));cap.set(1,native);ok,fr=cap.read();assert ok
   sx=fr.shape[1]/r['width'];sy=fr.shape[0]/r['height']
   for a,b in [(11,12),(11,23),(12,24),(23,24),(11,13),(13,15),(12,14),(14,16)]:
    if np.isfinite(xy[i,[a,b]]).all():cv2.line(fr,tuple((xy[i,a]*[sx,sy]).astype(int)),tuple((xy[i,b]*[sx,sy]).astype(int)),(0,0,255),2)
   im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB));im.thumbnail((195,375));can.paste(im,(j*200+(200-im.width)//2,rownum*430+42));dr.text((j*200+3,rownum*430+22),f'native {native} {native/nativefps:.2f}s',fill='black')
  cap.release();receipts.append({'id':r['id'],'sample_indices':idx.tolist(),'source_frames':[int(round(float(fs[i])/r['source_fps']*nativefps)) for i in idx],'source_sha256':r['source_sha256'],'cache_sha256':r['cache_sha256'],'page':f'base-train-{start//3+1:02d}.jpg'})
 can.save(O/f'base-train-{start//3+1:02d}.jpg',quality=94)
(O/'visual-preparation.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n');print(len(selected),'sources',len(range(0,len(selected),3)),'pages')
