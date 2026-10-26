import json,hashlib,cv2
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/paired-failure-mining-20261024';current=B/'output/paired-enriched-five-v2-20261024';rs=json.loads((current/'score-inputs.frozen.json').read_text())+json.loads((current/'test-inputs.frozen.json').read_text());roles={r['person']:r['split'] for r in rs if r['family']!='R2D2'};known={s for r in rs for s in r['sources']}|{r['source_sha256'] for r in json.loads((B/'output/ai-rule-paired-20261024/new-intake.json').read_text())};protected=set(json.loads((B/'output/five-exercise-recovery-20261023/final_test_reservation.json').read_text())['reserved_people'])|{'qevd_'+str(p) for p in json.loads((B/'data/qevd-selected/manifests/official_test_workers.json').read_text())}|{'qevd_2785','qevd_956'};pool=json.loads((B/'output/overnight-20261020/score_candidates.json').read_text());old=json.loads((O/'intake-v2.frozen.json').read_text())['records'];known|={r['source_sha256'] for r in old};people=set();new=[]
for ex in ['ytw','shoulder']:
 n=0
 for r in sorted(pool,key=lambda r:hashlib.sha256(('upper-review-v1'+r['id']).encode()).hexdigest()):
  if r['exercise']!=ex or r['split']!='train' or roles.get(r['person'],'train')!='train' or r['person'] in protected or r['person'] in people:continue
  if ex=='ytw' and 'arms too low' not in ' '.join(r['source_labels']):continue
  if ex=='shoulder' and ('/front/' not in r['path'] or 'bad' not in r['source_labels']):continue
  p=Path(r['path'])
  if not p.exists():continue
  h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
  if h in known:continue
  known.add(h);people.add(r['person']);new.append(dict(r,review_id=f'U{len(new)+1:03}',source_sha256=h,provider_tag_is_not_a_target=True));n+=1
  if n==4:break
for k,r in enumerate(new):
 c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);fps=c.get(5);N=int(c.get(7));r.update(fps=fps,frame_count=N,duration=N/fps)
 if k%3==0:canvas=Image.new('RGB',(1600,810),'white')
 fs=np.linspace(6.5*fps,15.25*fps,8).round().astype(int) if r['exercise']=='shoulder' else np.linspace(0,N-2,8).round().astype(int)
 for j,f in enumerate(fs):
  c.set(1,int(f));ok,a=c.read();assert ok;im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((195,240));x=j*200;y=k%3*270;canvas.paste(im,(x,y+25));ImageDraw.Draw(canvas).text((x+2,y+3),f"{r['review_id']} {f/fps:.2f}s",fill='black')
 c.release();canvas.save(O/f'upper-{k//3+1:02}.jpg',quality=94)
(O/'upper-intake.frozen.json').write_text(json.dumps(new,indent=2));print('UPPER',len(new),flush=True)
