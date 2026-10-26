"""Make per-video whole-length screens only after its last frame is available."""
import json,sys
from pathlib import Path
import cv2
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];O=B/'output/shoulder-common-support-audit-20261024/full-review-v1'
def main():
 inv=json.loads((B/'output/shoulder-work-20261024/full-review-v1/inventory.frozen.json').read_text())
 created=[]
 for r in inv['sources']:
  if r['owner']!='model_evaluation':continue
  dest=O/(r['id']+'-screen.jpg')
  if dest.exists():continue
  folder=O/'frames'/r['id']
  if not folder.exists():continue
  c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);N=int(c.get(cv2.CAP_PROP_FRAME_COUNT));fps=c.get(cv2.CAP_PROP_FPS);c.release()
  if not (folder/f'f{N-1:06}.jpg').exists():continue
  files=sorted(folder.glob('f*.jpg'));nums=[int(p.stem[1:]) for p in files];chosen=[]
  for t in range(int(N/fps)+1):
   j=min(range(len(nums)),key=lambda j:abs(nums[j]/fps-t));chosen.append((nums[j],files[j]))
  chosen=sorted(set(chosen));cols=8;w=170;h=245;out=Image.new('RGB',(cols*w,((len(chosen)+cols-1)//cols)*h+30),'white');d=ImageDraw.Draw(out)
  d.text((3,4),r['id']+' / '+r['split']+' / whole1Hz SCREEN ONLY',fill='black')
  for j,(f,p) in enumerate(chosen):
   im=Image.open(p);im.thumbnail((w,h-25));x=j%cols*w;y=j//cols*h+30;out.paste(im,(x+(w-im.width)//2,y+24));d.text((x+2,y+3),f'f{f} {f/fps:.2f}s',fill='black')
  out.save(dest,quality=95);created.append(r['id'])
 print(json.dumps(dict(created=len(created),ids=created)))
if __name__=='__main__':main()
