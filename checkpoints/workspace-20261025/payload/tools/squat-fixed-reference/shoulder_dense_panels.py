"""Prepare event-level5Hz windows for manual review; never assign grades."""
import json
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];O=B/'output/shoulder-common-support-audit-20261024/full-review-v1'
def main():
 rows=json.loads((O/'decoded-manifest.frozen.json').read_text())['sources'];outrows=[]
 warm={('bad','diagonal',1):14,('bad','diagonal',2):6,('bad','diagonal',7):2,
       ('bad','front',1):4,('bad','front',2):6,('bad','front',7):1.5,
       ('bad','side',1):3.5,('bad','side',2):5}
 for r in rows:
  n=int(r['person'][-3:]);start=warm.get((r['provider_tag'],r['view'],n),0)
  spans=[(start,min(start+4,r['duration'])),(max(start+4,r['duration']-2),r['duration'])]
  frames=[f for f in r['frames'] if any(a<=f['time']<=b for a,b in spans)]
  # Retain every5Hz evidence frame; pages never discard overshoot/end context.
  pages=[]
  for page,st in enumerate(range(0,len(frames),32),1):
   p=O/(r['id']+f'-dense-{page:02}.jpg');assert not p.exists();pages.append(str(p.relative_to(O)))
   out=Image.new('RGB',(1600,4*280+30),'white');d=ImageDraw.Draw(out)
   d.text((5,5),r['id']+' '+r['split']+' 5Hz original source pixels / selected early+end windows',fill='black')
   for j,f in enumerate(frames[st:st+32]):
    im=Image.open(O/f['path']);x=j%8*200;y=j//8*280+30;out.paste(im,(x+(200-im.width)//2,y+25));d.text((x+3,y+3),f"f{f['frame']} {f['time']:.2f}s",fill='black')
   out.save(p,quality=96)
  outrows.append(dict(id=r['id'],intervals=spans,frames=frames,sheets=pages,grades_not_assigned=True))
 p=O/'dense-panel-manifest.frozen.json';assert not p.exists();p.write_text(json.dumps(dict(records=outrows),indent=2));print('prepared',len(outrows),'sources',sum(len(r['sheets']) for r in outrows),'sheets')
if __name__=='__main__':main()
