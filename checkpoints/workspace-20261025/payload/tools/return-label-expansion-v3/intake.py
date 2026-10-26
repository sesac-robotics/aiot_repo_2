"""Freeze genuinely unused Train originals before visual review; hints are not labels."""
import json,sys,hashlib,collections
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
B=Path(__file__).resolve().parents[2]; O=B/'output/return-label-expansion-v3-20261024'
sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import assert_training_allowed,POLICY
read=lambda p:json.loads(p.read_text())
parent=B/'output/return-initial-context-20261024'
rows=read(parent/'score-inputs.frozen.json')+read(parent/'test-inputs.frozen.json')
roles=collections.defaultdict(set)
for r in rows:
 if r['family']!='R2D2': roles[r['person']].add(r['split'])
known={s for r in rows for s in r['sources']}
def collect(x):
 if isinstance(x,dict):
  for k,v in x.items():
   if k=='source_sha256' and isinstance(v,str):known.add(v)
   elif k=='sources' and isinstance(v,list):known.update(s for s in v if isinstance(s,str))
   else:collect(v)
 elif isinstance(x,list):
  for v in x:collect(v)
for p in sorted((B/'output').glob('**/*intake*.json')):
 if O in p.parents:continue
 collect(read(p))
protected=set(read(B/'output/five-exercise-recovery-20261023/final_test_reservation.json')['reserved_people'])|{'qevd_'+str(p) for p in read(B/'data/qevd-selected/manifests/official_test_workers.json')}|{'qevd_2785','qevd_956'}
def allowed(person):
 if person in protected:return False
 if person in roles:return roles[person]=={'train'}
 return int(hashlib.sha256(('return-label-v3-'+person).encode()).hexdigest()[:8],16)%5!=0
pool=[]
for r in read(B/'mediapipe-pilot/qevd_coach/catalog.json')['items']:
 person='qevd_'+str(r['worker'])
 if r['exercise']!='pushups' or r['split']!='train' or not allowed(person):continue
 cp=B/'mediapipe-pilot/qevd_coach/artifacts_v2/heavy/features'/(r['id']+'.npz')
 if not cp.exists():continue
 seq=np.load(cp)['sequence'];tops=[]
 for a,b,c in [(1,3,5),(2,4,6)]:
  u=seq[:,a]-seq[:,b];v=seq[:,c]-seq[:,b];den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1);ok=np.isfinite(seq[:,[a,b,c]]).all((1,2))&(den>1e-7)
  if ok.mean()<.9:continue
  tops.append(float(np.percentile(np.degrees(np.arccos(np.clip(np.sum(u[ok]*v[ok],1)/den[ok],-1,1))),95)))
 if len(tops)==2:pool.append(dict(r,person=person,priority=180-max(tops),hint='cached_low_top_extension_priority_only_not_grade'))
for r in read(B/'output/squat-errors-20261023/pose-search.json'):
 if allowed(r['person']) and r.get('torso_coverage',0)>=.9 and r['frame_count']>=160:
  pool.append(dict(r,exercise='squats',priority=r.get('return_signal') or 0,hint='cached_height_residual_priority_only_not_grade'))
chosen=[];people=set()
for exercise,count in [('pushups',12),('squats',8)]:
 for r in sorted((r for r in pool if r['exercise']==exercise),key=lambda r:(-r['priority'],r['id'])):
  if sum(a['exercise']==exercise for a in chosen)>=count:break
  if r['person'] in people or not Path(r['path']).exists():continue
  digest=sha(r['path'])
  if digest in known:continue
  c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);N=int(c.get(7));fps=float(c.get(5));c.release()
  if N<40 or not fps>0:continue
  a=dict(r,review_id=f'E{len(chosen)+1:02}',source_sha256=digest,sources=[digest],split='train',fps=fps,frame_count=N,role_frozen_before_review=True)
  assert_training_allowed([a]);chosen.append(a);people.add(a['person']);known.add(digest)
protocol=dict(parent_run=str(parent),policy_sha256=sha(POLICY),selection_uses_Test_results=False,final_Test_opened=False,source_role_policy='existing external person role or deterministic pre-review role; Train only; official Test workers excluded',labels_not_yet_assigned=True,records=chosen)
p=O/'intake.frozen.json';assert not p.exists();p.write_text(json.dumps(protocol,indent=2))
for r in chosen:
 c=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);frames=np.linspace(0,r['frame_count']-3,12).round().astype(int);canvas=Image.new('RGB',(1800,560),'white');draw=ImageDraw.Draw(canvas)
 for j,f in enumerate(frames):
  c.set(1,int(f));ok,a=c.read();assert ok;im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((300,245));x=j%6*300;y=j//6*280;canvas.paste(im,(x+(300-im.width)//2,y+25));draw.text((x,y),f"{r['review_id']} {r['person']} f{f} {f/r['fps']:.2f}s",fill='black')
 c.release();canvas.save(O/(r['review_id']+'-overview.jpg'),quality=94);print(r['review_id'],r['exercise'],r['person'],r['id'],flush=True)
print('intake',len(chosen),'people',len(people),flush=True)
