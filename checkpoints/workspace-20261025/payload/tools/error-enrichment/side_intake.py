"""Mine existing native pose cache to FIND review candidates, never generate labels."""
from pathlib import Path
import sys,json,collections,hashlib,cv2,numpy as np,re
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/error-enrichment-20261024';cat=json.loads((B/'mediapipe-pilot/qevd_coach/catalog.json').read_text())['items'];old=json.loads((O/'score-inputs.frozen.json').read_text());known={n for r in old for n in re.findall(r'\d{8}',r['id'])};roles={r['person']:r['split'] for r in old};protected=set(json.loads((B/'output/five-exercise-recovery-20261023/final_test_reservation.json').read_text())['reserved_people'])|{'qevd_'+str(p) for p in json.loads((B/'data/qevd-selected/manifests/official_test_workers.json').read_text())};candidates=[]
for r in cat:
 if r['exercise'] not in ['plank','pushups'] or r['split']!='train' or 'qevd_'+r['worker'] in protected or r['id'] in known:continue
 p=B/'mediapipe-pilot/qevd_coach/artifacts_v2/heavy/features'/(r['id']+'.npz')
 if not p.exists():continue
 z=np.load(p);seq=z['sequence'];side=[]
 for s,h,k,a in [(1,7,9,11),(2,8,10,12)]:
  good=np.isfinite(seq[:,[s,h,k,a]]).all((1,2));dx=np.abs(seq[:,a,0]-seq[:,s,0]);dz=np.abs(seq[:,a,2]-seq[:,s,2]);side.append((float(good.mean()),float(np.nanmedian(dx[good]/np.maximum(dz[good],.05))) if good.any() else 0))
 coverage,signal=max(side,key=lambda x:x[0]*min(x[1],5))
 if coverage<.7 or signal<.8:continue
 candidates.append(dict(r,selection_coverage=coverage,selection_lateral_signal=signal,not_a_target=True))
selected=[];people=set()
for ex,tags in [('plank',['high hips','butt sinking','REFERENCE']),('pushups',['elevated hips','shallow','REFERENCE'])]:
 for tag in tags:
  rs=[r for r in candidates if r['exercise']==ex and (r['reference'] if tag=='REFERENCE' else tag in r['errors'])];rs.sort(key=lambda r:-min(r['selection_lateral_signal'],5)*r['selection_coverage']);n=0
  for r in rs:
   person='qevd_'+r['worker']
   if person in people:continue
   r=dict(r);r.update(review_id='S'+str(len(selected)+1).zfill(3),person=person,split=roles.get(person,'train'),sha256=hashlib.file_digest(Path(r['path']).open('rb'),'sha256').hexdigest());selected.append(r);people.add(person);n+=1
   if n==4:break
for k,r in enumerate(selected):
 c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);fps=c.get(5);N=int(c.get(7));r.update(fps=fps,frame_count=N,duration=N/fps);fs=np.linspace(0,N-2,8).round().astype(int);r['review_frames']=fs.tolist()
 if k%4==0:canvas=Image.new('RGB',(1600,4*230),'#dddddd')
 for j,f in enumerate(fs):
  c.set(1,int(f));ok,a=c.read();assert ok
  im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((200,205));x=j*200;y=(k%4)*230;canvas.paste(im,(x+(200-im.width)//2,y+23));ImageDraw.Draw(canvas).text((x+2,y+3),r['review_id']+' '+str(round(f/fps,2))+'s',fill='black')
 c.release();r['sheet']=str(O/f'side-review-{k//4+1:02}.jpg');canvas.save(r['sheet'],quality=92)
(O/'side-intake.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2));(O/'side-screen.json').write_text(json.dumps(dict(candidates=len(candidates),selected=len(selected),selection_only_not_targets=True),indent=2));print('side candidates',len(candidates),'selected',len(selected),flush=True)
