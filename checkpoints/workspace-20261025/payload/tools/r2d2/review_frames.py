from pathlib import Path
import subprocess,json,time
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/r2d2-review-20261024';S=Path('/home/jayden/Downloads/R2D2');D=B/'data/r2d2-team-20261023/raw'
for n,src in enumerate(sorted(S.glob('*.mp4')),1):
 p=D/src.name
 while not p.exists() or p.stat().st_size!=src.stat().st_size:time.sleep(2)
 vid=f'V{n:02}';out=O/'dense-frames'/vid;out.mkdir(parents=True,exist_ok=True)
 if not (out/'done.json').exists():
  subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(p),'-an','-vf','fps=4,scale=240:426:force_original_aspect_ratio=decrease,pad=240:426:(ow-iw)/2:(oh-ih)/2','-q:v','3',str(out/'%04d.jpg')],check=True)
  (out/'done.json').write_text(json.dumps({'fps':4,'boundary_resolution_seconds':.25,'review_method':'direct_chronological_frame_review_not_continuous_playback'}))
 frames=sorted(out.glob('*.jpg'))
 for start in range(0,len(frames),48):
  fs=frames[start:start+48];im=Image.new('RGB',(8*240,((len(fs)+7)//8)*448),'white');draw=ImageDraw.Draw(im)
  for k,f in enumerate(fs):
   x=(k%8)*240;y=(k//8)*448
   draw.text((x+4,y+3),f'{vid} {(start+k)/4:.2f}s',fill='black');im.paste(Image.open(f),(x,y+20))
  im.save(O/f'{vid}-dense-{start//48+1:02}.jpg',quality=88)
 print(vid,len(frames),flush=True)
