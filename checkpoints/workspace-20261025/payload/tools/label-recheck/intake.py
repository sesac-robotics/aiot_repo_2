from pathlib import Path
import json,collections,hashlib,cv2,numpy as np
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/label-recheck-20261024';pool=json.loads((B/'output/overnight-20261020/score_candidates.json').read_text());old=json.loads((B/'output/label-verified-five-20261024/score-inputs.frozen.json').read_text())+json.loads((B/'output/label-verified-five-20261024/test-inputs.frozen.json').read_text());roles={r['person']:r['split'] for r in old};reserved=set(json.loads((B/'output/five-exercise-recovery-20261023/final_test_reservation.json').read_text())['reserved_people'])|{'qevd_'+str(p) for p in json.loads((B/'data/qevd-selected/manifests/official_test_workers.json').read_text())};known={s for r in old for s in r['sources']};selected=[]
def take(ex,term,view,n):
 people=set()
 for r in sorted(pool,key=lambda r:r['id']):
  if r['exercise']!=ex or r['split']!='train' or r['person'] in reserved or roles.get(r['person'],'train')!='train' or r['person'] in people:continue
  if term not in ' '.join(r['source_labels']):continue
  if view and not r['id'].endswith('_'+view):continue
  p=Path(r['path'])
  if not p.exists():continue
  digest=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
  if digest in known:continue
  known.add(digest);people.add(r['person']);selected.append(dict(r,review_id=f'N{len(selected)+1:03}',source_sha256=digest,selection_hint_not_label=True))
  if len(people)>=n:break
for args in [('ytw','arms too low',None,6),('ytw','torso parallel to the ground, thumbs pointed up',None,6),('pushups','bad','side',5),('pushups','good','side',5),('shoulder','bad','side',6)]:take(*args)
for k,r in enumerate(selected):
 c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);fps=c.get(5);N=int(c.get(7));r.update(fps=fps,frame_count=N,duration=N/fps);fs=np.linspace(0,N-2,10).round().astype(int)
 if k%4==0:canvas=Image.new('RGB',(1600,4*260),'#dddddd')
 for j,f in enumerate(fs):
  c.set(1,int(f));ok,a=c.read();assert ok
  tile=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));tile.thumbnail((160,230));x=j*160;y=(k%4)*260;canvas.paste(tile,(x+(160-tile.width)//2,y+25));ImageDraw.Draw(canvas).text((x+2,y+3),r['review_id']+' '+str(round(f/fps,2))+'s',fill='black')
 c.release();r['sheet']=str(O/f'new-review-{k//4+1:02}.jpg');canvas.save(r['sheet'],quality=91)
(O/'new-intake.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2));print('NEW_TRAINING_SOURCES',len(selected),collections.Counter(r['exercise'] for r in selected),flush=True)
