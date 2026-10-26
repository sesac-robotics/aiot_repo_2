import json,sys,cv2
from pathlib import Path
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2];O=B/'output/mild-return-mining-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
read=lambda p:json.loads(p.read_text());old=B/'output/return-root-cause-v3-20261024';rows={r['id']:r for r in read(old/'score-inputs.frozen.json')};labs={r['id']:r for r in read(B/'output/squat-return-20261023/labels-v2.frozen.json')['labels']};team={r['id']:r for r in read(B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json')['labels']};inv={r['id']:r for r in read(B/'output/r2d2-review-20261024/inventory.json')};refs=read(B/'output/squat-side-return-v5-20261024/MODEL_SELECTION_PROTOCOL.frozen.json')['reference_windows'];rs=[]
for a in read(old/'feature-provenance.json'):
 if not (a['split']=='train' and a['grade']==2 and a['eligible'] and a['features'][0]<.16 and .06<a['features'][3]<.35):continue
 r=rows[a['id']]
 if r['family']=='R2D2':
  d=team[r['id'][5:]];src=inv[d['source_id']];fps=eval(src['fps'],{'__builtins__':{}});reference_frame=round(sum(refs[d['source_id']])/2*fps);path=src['copy']
 else:
  d=labs[r['id'][7:]];srcs=[x for x in labs.values() if x['path']==d['path'] and x['grade']==2];reference_frame=21 if 'diag_m4' in r['id'] else min(x['last_frame'] for x in srcs)-3;fps=d['fps'];path=d['path']
 rs.append(dict(a,review_id=f'C{len(rs)+1:02}',person=r['person'],source_sha256=r['sources'][0],path=path,reference_frame=reference_frame,fps=fps,review_frames=[reference_frame]+[max(0,a['return_source_frame']+k) for k in [-9,-3,3,9,15,21,27]],selection='Train-only recheck of mild hip/shoulder residuals; not predictions or Test outcomes',existing_grade_is_not_review_decision=True))
assert len(rs)==32 and not (O/'intake.frozen.json').exists();(O/'intake.frozen.json').write_text(json.dumps(rs,indent=2))
for k,r in enumerate(rs):
 if k%4==0:canvas=Image.new('RGB',(8*220,4*300),'white')
 draw=ImageDraw.Draw(canvas);row=k%4;y=row*300;draw.text((2,y),r['review_id']+' '+r['id'],fill='black');cap=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);N=int(cap.get(7));r['review_frames']=[min(f,N-1) for f in r['review_frames']]
 for j,f in enumerate(r['review_frames']):
  cap.set(1,f);ok,a=cap.read();assert ok
  import numpy as np
  cols=np.flatnonzero((a.max(2)>20).mean(0)>.1)
  if len(cols):a=a[:,cols[0]:cols[-1]+1]
  im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((220,248));canvas.paste(im,(j*220+(220-im.width)//2,y+24));draw.text((j*220+2,y+275),('REF ' if j==0 else '')+f'f{f} {f/r["fps"]:.2f}s',fill='black')
 cap.release();canvas.save(O/f'review-{k//4+1:02}.jpg',quality=94);print('PREVIEW',r['review_id'],flush=True)
