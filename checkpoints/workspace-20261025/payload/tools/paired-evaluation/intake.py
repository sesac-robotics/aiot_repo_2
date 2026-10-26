"""Mine native Train only; provider hints select review candidates, never grades."""
import json,collections,hashlib,cv2,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/ai-rule-paired-20261024';base=B/'output/pushup-repaired-five-v2-20261024';current=json.loads((base/'score-inputs.frozen.json').read_text())+json.loads((base/'test-inputs.frozen.json').read_text());roles={r['person']:r['split'] for r in current if not r['person'].startswith('r2d2_')};protected=set(json.loads((B/'output/five-exercise-recovery-20261023/final_test_reservation.json').read_text())['reserved_people'])|{'qevd_'+str(p) for p in json.loads((B/'data/qevd-selected/manifests/official_test_workers.json').read_text())}|{'qevd_2785','qevd_956'};known={s for r in current for s in r['sources']};pool=json.loads((B/'output/overnight-20261020/score_candidates.json').read_text());selected=[];people=set()
for ex,tag,n in [('pushups','insufficient',5),('pushups','early stopping',5),('pushups','legs on the floor',4),('pushups','shallow',4),('shoulder','bad',4)]:
 found=0
 for r in sorted(pool,key=lambda r:hashlib.sha256(('paired-v1'+r['id']).encode()).hexdigest()):
  if r['exercise']!=ex or r['split']!='train' or r['person'] in protected or roles.get(r['person'],'train')!='train' or r['person'] in people:continue
  if tag not in ' '.join(r['source_labels']):continue
  p=Path(r['path'])
  if not p.exists():continue
  digest=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
  if digest in known:continue
  known.add(digest);people.add(r['person']);selected.append(dict(r,review_id=f'P{len(selected)+1:03}',source_sha256=digest,selection_hint_not_label=tag));found+=1
  if found>=n:break
for k,r in enumerate(selected):
 c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);fps=c.get(5);N=int(c.get(7));r.update(fps=fps,frame_count=N,duration=N/fps)
 if k%4==0:canvas=Image.new('RGB',(1600,4*260),'white')
 for j,f in enumerate(np.linspace(0,N-2,10).round().astype(int)):
  c.set(1,int(f));ok,a=c.read();assert ok;tile=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));tile.thumbnail((160,230));x=j*160;y=k%4*260;canvas.paste(tile,(x+(160-tile.width)//2,y+25));ImageDraw.Draw(canvas).text((x+2,y+3),f"{r['review_id']} {f/fps:.2f}s",fill='black')
 c.release();r['sheet']=str(O/f'intake-{k//4+1:02}.jpg');canvas.save(r['sheet'],quality=91)
(O/'new-intake.json').write_text(json.dumps(selected,indent=2));print('NATIVE_TRAIN_REVIEW',len(selected),collections.Counter(r['exercise'] for r in selected),flush=True)
