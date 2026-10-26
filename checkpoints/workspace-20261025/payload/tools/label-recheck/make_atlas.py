from pathlib import Path
import json,math
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/label-recheck-20261024';rows=json.loads((B/'data/r2d2-team-20261023/annotations/labels.frozen.json').read_text())['labels'];manifest=[]
for ex in ['squats','shoulder','pushups','ytw','plank']:
 rs=[r for r in rows if r['exercise']==ex]
 for start in range(0,len(rs),6):
  subset=rs[start:start+6];im=Image.new('RGB',(9*150,len(subset)*268),'white');draw=ImageDraw.Draw(im)
  for j,r in enumerate(subset):
   fs=sorted((B/'output/r2d2-review-20261024/dense-frames'/r['source_id']).glob('*.jpg'));n=len(fs);times=[r['start']+(r['end']-r['start'])*f for f in [.02,.14,.26,.38,.5,.62,.74,.86,.98]];samples=[]
   for k,t in enumerate(times):
    index=max(0,min(n-1,round(t*4)));p=fs[index];a=Image.open(p);a.thumbnail((150,244));x=k*150;y=j*268;im.paste(a,(x+(150-a.width)//2,y+23));draw.text((x+2,y+3),f'{r["id"][:3]} {t:.2f}',fill='black');samples.append(dict(path=str(p),approx_time=index/4))
   manifest.append(dict(id=r['id'],exercise=ex,source_id=r['source_id'],start=r['start'],end=r['end'],sheet=str(O/f'{ex}-{start//6+1:02}.jpg'),row=j,samples=samples,prior_label_or_prediction_displayed=False))
  im.save(O/f'{ex}-{start//6+1:02}.jpg',quality=91)
(O/'atlas-manifest.json').write_text(json.dumps(manifest,indent=2));print('ATLAS',len(rows),len(set(r['sheet'] for r in manifest)),flush=True)
