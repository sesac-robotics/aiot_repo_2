import json,hashlib,cv2,numpy as np
from pathlib import Path
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/paired-failure-mining-20261024';P=B/'output/paired-enriched-five-v2-20261024';rows=json.loads((P/'score-inputs.frozen.json').read_text())+json.loads((P/'test-inputs.frozen.json').read_text());roles={r['person']:r['split'] for r in rows if r['family']!='R2D2'};known={s for r in rows for s in r['sources']};protected=set(json.loads((B/'output/five-exercise-recovery-20261023/final_test_reservation.json').read_text())['reserved_people'])|{'qevd_'+str(p) for p in json.loads((B/'data/qevd-selected/manifests/official_test_workers.json').read_text())}|{'qevd_2785','qevd_956'}
for file,key in [(B/'output/ai-rule-paired-20261024/new-intake.json',None),(O/'intake-v2.frozen.json','records'),(O/'upper-intake.frozen.json',None)]:
 x=json.loads(file.read_text());x=x[key] if key else x;known|={r['source_sha256'] for r in x}
pool=[]
for r in json.loads((B/'mediapipe-pilot/qevd_coach/catalog.json').read_text())['items']:
 person='qevd_'+str(r['worker']);tag=' '.join(r.get('errors',[]))
 if r['exercise']!='plank' or r['split']!='train' or roles.get(person,'train')!='train' or person in protected or not any(t in tag for t in ['on knees','legs and hips on the floor','stopping early']):continue
 p=B/'mediapipe-pilot/qevd_coach/artifacts_v2/heavy/features'/(r['id']+'.npz')
 if not p.exists():continue
 seq=np.load(p)['sequence'];signals=[]
 for sh,hip,knee,ankle in [(1,7,9,11),(2,8,10,12)]:
  good=np.isfinite(seq[:,[sh,hip,knee,ankle]]).all((1,2));dx=np.abs(seq[:,ankle,0]-seq[:,sh,0]);dz=np.abs(seq[:,ankle,2]-seq[:,sh,2]);signal=float(np.nanmedian(dx[good]/np.maximum(dz[good],.05))) if good.any() else 0
  if good.mean()>=.8 and signal>=.8:signals.append(min(signal,5)*float(good.mean()))
 if signals:pool.append(dict(r,person=person,lateral_selection_signal=max(signals),selection_not_a_label=True))
new=[];people=set()
for r in sorted(pool,key=lambda r:-r['lateral_selection_signal']):
 if r['person'] in people:continue
 p=Path(r['path'])
 if not p.exists():continue
 h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
 if h in known:continue
 known.add(h);people.add(r['person']);new.append(dict(r,review_id=f'L{len(new)+1:03}',source_sha256=h))
 if len(new)==4:break
for k,r in enumerate(new):
 c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);fps=c.get(5);N=int(c.get(7));r.update(fps=fps,frame_count=N,duration=N/fps)
 if k%2==0:canvas=Image.new('RGB',(1600,540),'white')
 for j,f in enumerate(np.linspace(0,N-2,8).round().astype(int)):
  c.set(1,int(f));ok,a=c.read();assert ok;im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((195,240));x=j*200;y=k%2*270;canvas.paste(im,(x,y+25));ImageDraw.Draw(canvas).text((x+2,y+3),f"{r['review_id']} {f/fps:.2f}s",fill='black')
 c.release();canvas.save(O/f'lateral-{k//2+1:02}.jpg',quality=94)
(O/'lateral-intake.frozen.json').write_text(json.dumps(new,indent=2));print('LATERAL',len(pool),len(new),flush=True)
