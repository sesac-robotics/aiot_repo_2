"""Train-only cache mining to locate visible failure review candidates, never labels."""
import json,sys,hashlib,collections,cv2,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/paired-failure-mining-20261024';O.mkdir(exist_ok=True);P=B/'output/paired-enriched-five-v2-20261024';rows=json.loads((P/'score-inputs.frozen.json').read_text())+json.loads((P/'test-inputs.frozen.json').read_text());roles={r['person']:r['split'] for r in rows if r['family']!='R2D2'};protected=set(json.loads((B/'output/five-exercise-recovery-20261023/final_test_reservation.json').read_text())['reserved_people'])|{'qevd_'+str(p) for p in json.loads((B/'data/qevd-selected/manifests/official_test_workers.json').read_text())}|{'qevd_2785','qevd_956'};known={s for r in rows for s in r['sources']}|{r['source_sha256'] for r in json.loads((B/'output/ai-rule-paired-20261024/new-intake.json').read_text())};cat=json.loads((B/'mediapipe-pilot/qevd_coach/catalog.json').read_text())['items'];pool=[]
for r in cat:
 person='qevd_'+str(r['worker'])
 if r['split']!='train' or roles.get(person,'train')!='train' or person in protected or r['exercise'] not in ['pushups','plank','ytw','squats']:continue
 p=B/'mediapipe-pilot/qevd_coach/artifacts_v2/heavy/features'/(r['id']+'.npz')
 if not p.exists():continue
 z=np.load(p);seq=z['sequence']
 if seq.ndim!=3 or seq.shape[1:]!=(17,3):continue
 tag=' '.join(r.get('errors',[]));hint=None;priority=0.
 if r['exercise']=='pushups':
  angles=[]
  for a,b,c in [(1,3,5),(2,4,6)]:
   u=seq[:,a]-seq[:,b];v=seq[:,c]-seq[:,b];angle=np.degrees(np.arccos(np.clip(np.sum(u*v,1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)+1e-9),-1,1)));good=np.isfinite(seq[:,[a,b,c]]).all((1,2));angle=angle[good]
   if len(angle)>=8 and len(angle)>=.8*len(seq):angles.append(float(np.percentile(angle,90)))
  if len(angles)==2 and max(angles)<160:hint='cache_low_arm_extension_not_a_label';priority=160-max(angles)
 elif r['exercise']=='plank' and any(t in tag for t in ['on knees','legs and hips on the floor','stopping early']):hint='support_or_interruption_provider_hint_not_a_label';priority=1.
 elif r['exercise']=='ytw' and 'arms too low' in tag:hint='low_Y_provider_hint_not_a_label';priority=1.
 elif r['exercise']=='squats' and any(t in tag for t in ['shallow','insufficient']):hint='depth_provider_hint_not_a_label';priority=1.
 if hint:pool.append(dict(r,person=person,selection_hint=hint,priority=priority,cache=str(p)))
selected=[];people=set()
for ex,n in [('pushups',8),('plank',4),('ytw',3),('squats',3)]:
 found=0
 for r in sorted([r for r in pool if r['exercise']==ex],key=lambda r:(-r['priority'],hashlib.sha256(('review-failure-v1'+r['id']).encode()).hexdigest())):
  if r['person'] in people:continue
  path=Path(r['path'])
  if not path.exists():continue
  digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
  if digest in known:continue
  known.add(digest);people.add(r['person']);selected.append(dict(r,review_id=f'F{len(selected)+1:03}',source_sha256=digest));found+=1
  if found>=n:break
for k,r in enumerate(selected):
 c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);fps=c.get(5);N=int(c.get(7));r.update(fps=fps,frame_count=N,duration=N/fps)
 if k%3==0:canvas=Image.new('RGB',(1600,3*270),'white')
 for j,f in enumerate(np.linspace(0,N-2,8).round().astype(int)):
  c.set(1,int(f));ok,a=c.read();assert ok;im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((195,240));x=j*200;y=k%3*270;canvas.paste(im,(x,y+25));ImageDraw.Draw(canvas).text((x+2,y+3),f"{r['review_id']} {f/fps:.2f}s",fill='black')
 c.release();canvas.save(O/f'review-{k//3+1:02}.jpg',quality=94)
(O/'intake-v2.frozen.json').write_text(json.dumps(dict(selection_not_labels=True,train_only=True,counts=dict(collections.Counter(r['exercise'] for r in selected)),records=selected),indent=2));print('MINED',len(pool),'SELECTED',len(selected),flush=True)
