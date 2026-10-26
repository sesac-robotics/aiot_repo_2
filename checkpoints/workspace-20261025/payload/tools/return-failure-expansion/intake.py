"""Freeze unseen Train/development candidates; geometric hints are not labels."""
import json,sys,hashlib,collections
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/return-failure-expansion-20261024';O.mkdir(exist_ok=True)
sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
read=lambda p:json.loads((B/p).read_text())
parent=read('output/paired-failure-enriched-five-v2-20261024/score-inputs.frozen.json');roles={r['person']:r['split'] for r in parent if r['family']!='R2D2'};known={s for r in parent for s in r['sources']}
for name in ['output/paired-failure-mining-20261024/labels-v2.frozen.json','output/ai-rule-paired-20261024/new-intake.json']:
 a=read(name);a=a['reviewed'] if isinstance(a,dict) else a;known|={r.get('source_sha256') for r in a}
protected=set(read('output/five-exercise-recovery-20261023/final_test_reservation.json')['reserved_people'])|{'qevd_'+str(p) for p in read('data/qevd-selected/manifests/official_test_workers.json')}|{'qevd_2785','qevd_956'}
cat=read('mediapipe-pilot/qevd_coach/catalog.json')['items'];pool=[]
for r in cat:
 person='qevd_'+str(r['worker'])
 if r['split']!='train' or person in protected or roles.get(person,'train')!='train' or r['exercise']!='pushups':continue
 cp=B/'mediapipe-pilot/qevd_coach/artifacts_v2/heavy/features'/(r['id']+'.npz')
 if not cp.exists():continue
 seq=np.load(cp)['sequence'];angles=[]
 for a,b,c in [(1,3,5),(2,4,6)]:
  u=seq[:,a]-seq[:,b];v=seq[:,c]-seq[:,b];den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1);ok=np.isfinite(seq[:,[a,b,c]]).all((1,2))&(den>1e-7)
  if ok.mean()<.8:continue
  angles.append(float(np.percentile(np.degrees(np.arccos(np.clip(np.sum(u[ok]*v[ok],1)/den[ok],-1,1))),90)))
 if len(angles)==2:pool.append(dict(r,person=person,priority=180-max(angles),selection_hint='low_cached_top_extension_is_review_priority_only'))
for r in read('output/squat-errors-20261023/pose-search.json'):
 if r['person'] in protected or roles.get(r['person'],'train')!='train' or r.get('torso_coverage',0)<.8 or (r.get('return_signal') or 0)<.5:continue
 pool.append(dict(r,exercise='squats',priority=r['return_signal'],selection_hint='possible_incomplete_height_return_is_review_priority_only'))
selected=[];people=set()
for ex,n in [('pushups',16),('squats',8)]:
 for r in sorted([r for r in pool if r['exercise']==ex],key=lambda r:(-r['priority'],r['id'])):
  if sum(s['exercise']==ex for s in selected)>=n:break
  if r['person'] in people or not Path(r['path']).exists():continue
  digest=sha(r['path'])
  if digest in known:continue
  role=roles.get(r['person'])
  if role is None:role='validation' if int(hashlib.sha256(('return-expansion-fixed-dev-'+r['person']).encode()).hexdigest()[:8],16)%5==0 else 'train'
  cap=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);fps=cap.get(5);N=int(cap.get(7));cap.release();assert fps>0 and N>5
  selected.append(dict(r,review_id=f'N{len(selected)+1:03}',source_sha256=digest,fps=fps,frame_count=N,duration=N/fps,split=role,role_frozen_before_visual_review=True));known.add(digest);people.add(r['person'])
assert not (O/'intake.frozen.json').exists();(O/'intake.frozen.json').write_text(json.dumps(dict(records=selected,candidates_not_training_labels=True,selection_uses_no_Test_results=True,final_Test_people_excluded=True),indent=2))
for k,r in enumerate(selected):
 cap=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);frames=np.linspace(0,r['frame_count']-2,10).round().astype(int);canvas=Image.new('RGB',(1700,240),'white');d=ImageDraw.Draw(canvas)
 for j,f in enumerate(frames):
  cap.set(1,int(f));ok,a=cap.read();assert ok;im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));im.thumbnail((170,200));canvas.paste(im,(j*170+(170-im.width)//2,25));d.text((j*170+2,2),f"{r['review_id']} {f/r['fps']:.2f}s",fill='black')
 cap.release();canvas.save(O/(r['review_id']+'-overview.jpg'),quality=92);print(r['review_id'],r['exercise'],r['split'],r['id'],flush=True)
print('INTAKE_COMPLETE',len(selected),dict(collections.Counter(r['split'] for r in selected)),flush=True)
