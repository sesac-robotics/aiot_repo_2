from pathlib import Path
import json,hashlib,sys,collections,cv2
import numpy as np
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-work-20261024/normal-reference-v2';R=B/'output/shoulder-work-20261024/full-review-v1';inv={r['id']:r for r in json.loads((R/'inventory.frozen.json').read_text())['sources']};rows=json.loads((O/'inputs.frozen.json').read_text());audit=[]
for r in rows:
 assert r['split']!='test';s=inv[r['source_id']];assert s['source_sha256']==r['source_sha256'] and s['split']==r['split'];p=Path(r['cache']);m=json.loads(p.with_suffix('.json').read_text());z=np.load(p);assert m['config']['source_sha256']==s['source_sha256'];assert str(z['fingerprint'])==m['fingerprint']==hashlib.sha256(json.dumps(m['config'],sort_keys=True).encode()).hexdigest()
 if r['item']=='lower_return':assert min(r['phase']['return_ticks'])>r['phase']['top_tick']
 audit.append(dict(id=r['id'],cache_verified=True,role_verified=True,production_eligible=False))
# Chain overlays only for inputs that are used by a fitted head (seated held head excluded).
seen=set();targets=[]
for r in rows:
 if r['bank']=='front_seated':continue
 key=(r['source_id'],r['cache'])
 if key in seen:continue
 seen.add(key);targets.append(r)
for page in range((len(targets)+3)//4):
 can=Image.new('RGB',(1200,1440),'white');dr=ImageDraw.Draw(can)
 for row,r in enumerate(targets[page*4:(page+1)*4]):
  s=inv[r['source_id']];p=Path(r['cache']);z=np.load(p);times=z['time_seconds'];tr=r.get('observed_time_range') or [min(r['phase']['reference_ticks'])/10,max(r['phase']['return_ticks'])/10];indices=np.where((times>=tr[0]-.1)&(times<=tr[1]+.1))[0];selected=indices[np.linspace(0,len(indices)-1,3).round().astype(int)];c=cv2.VideoCapture(s['path']);dr.text((4,row*360+2),r['id']+' / '+p.parent.name,fill='black')
  for col,j in enumerate(selected):
   c.set(1,int(z['frame_numbers'][j]));ok,fr=c.read();assert ok;h,w=fr.shape[:2];im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB));im.thumbnail((400,320));x=col*400+(400-im.width)//2;y=row*360+24;can.paste(im,(x,y));factor=im.width/w
   for a,b in [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24)]:
    xy=z['image_xy'][j,[a,b]]
    if np.isfinite(xy).all():dr.line([(x+q[0]*factor,y+q[1]*factor) for q in xy],fill='lime',width=2)
   dr.text((col*400+3,row*360+343),f'f{z["frame_numbers"][j]} t{times[j]:.3f}s',fill='black')
  c.release()
 can.save(O/f'chain-audit-{page+1:02d}.jpg',quality=90)
(O/'input-verification.json').write_text(json.dumps(dict(rows=audit,hash_role_fingerprint_verified=True,test_rows=0,visual_chain_audit_pending=True,overlay_pages=(len(targets)+3)//4),ensure_ascii=False,indent=2)+'\n');print('unique chains',len(targets),'pages',(len(targets)+3)//4)
