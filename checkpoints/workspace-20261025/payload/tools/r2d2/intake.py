from pathlib import Path
import subprocess,json,hashlib,shutil
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');S=Path('/home/jayden/Downloads/R2D2');D=B/'data/r2d2-team-20261023/raw';O=B/'output/r2d2-review-20261024';rows=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
for n,p in enumerate(sorted(S.glob('*.mp4')),1):
 dst=D/p.name
 if not dst.exists():shutil.copy2(p,dst)
 a=sha(p);b=sha(dst);assert a==b
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(dst)]));v=next(x for x in probe['streams'] if x['codec_type']=='video');duration=float(probe['format']['duration'])
 r=dict(id=f'V{n:02}',filename=p.name,source=str(p),copy=str(dst),sha256=b,bytes=dst.stat().st_size,duration=duration,width=v['width'],height=v['height'],fps=v['avg_frame_rate'],rotation=v.get('side_data_list',[]),streams=[x['codec_type'] for x in probe['streams']]);rows.append(r)
 # All frames are decoded sequentially by ffmpeg while sampling 6 overview images.
 tmp=O/'overview-frames'/r['id'];tmp.mkdir(parents=True,exist_ok=True)
 subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(dst),'-vf',f'fps=6/{duration},scale=320:240:force_original_aspect_ratio=decrease,pad=320:240:(ow-iw)/2:(oh-ih)/2','-frames:v','6',str(tmp/'%02d.jpg')],check=True)
 print(r['id'],p.name,round(duration,2),flush=True)
(O/'inventory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
for start in range(0,len(rows),8):
 group=rows[start:start+8];im=Image.new('RGB',(6*320,len(group)*268),'#fafafa');draw=ImageDraw.Draw(im)
 for y,r in enumerate(group):
  draw.text((4,y*268+3),f"{r['id']} {r['filename']} {r['duration']:.1f}s",fill='black')
  for i,p in enumerate(sorted((O/'overview-frames'/r['id']).glob('*.jpg'))):im.paste(Image.open(p),(i*320,y*268+25))
 im.save(O/f'overview-{start//8+1:02}.jpg',quality=90)
print('TOTAL',len(rows),sum(r['duration'] for r in rows))
