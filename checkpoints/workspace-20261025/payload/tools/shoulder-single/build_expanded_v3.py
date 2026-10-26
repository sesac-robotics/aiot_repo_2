"""Train-only reviewed windows; no label inferred from landmark values."""
from pathlib import Path
import json,sys,hashlib,numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-expanded-train-v3-20261025';sys.path.insert(0,str(B/'tools/shoulder-work'))
from observation_features_v4 import extract_body
from anatomical_turn_features_v1 import extract as lower
S={r['id']:r for r in json.loads((O/'source-plan.frozen.json').read_text())}
# peak and lower rack intervals come from source contact sheets, never pose maxima.
plan=[('V20',1.3,3.4,[1.4,1.7],[3.0,3.3],2),('V21',.3,2.0,[.4,.7],[1.6,1.9],2),('V24',1.4,3.4,[1.5,1.8],[3.0,3.3],2),('V25',1.2,2.9,None,None,2),('V26',1.2,3.9,[1.4,1.7],[3.5,3.8],2),('V27',1.3,3.7,[1.5,1.8],[3.3,3.6],2),('V27',17.9,19.7,None,None,1),('V27',20.2,22.0,None,None,1)]
rows=[]
for n,(sid,a,b,p,t,g) in enumerate(plan):
 r=S[sid];path=B/f'output/integrated-five-20261024/team-observations/{sid}/side.npz';m=json.loads(path.with_suffix('.json').read_text());z=np.load(path);assert m['config']['source_sha256']==r['source_sha256'];assert str(z['fingerprint'])==m['fingerprint'];assert hashlib.sha256(json.dumps(m['config'],sort_keys=True).encode()).hexdigest()==m['fingerprint'];assert abs(m['config']['segment']['fps']-r['fps'])<1e-6
 fs=z['frame_numbers'];times=fs/r['fps'];ticks=np.rint(times*10).astype(int);assert len(set(ticks))==len(ticks);mask=(ticks>=round(a*10))&(ticks<=round(b*10));ss=list(zip(ticks[mask],z['image_xy'][mask]));nf=fs[mask];body=extract_body(ss,[round(a*10),round(b*10)],True,r['width'],r['height'],'front');event=None;lo={'status':'held','reason':'nonbilateral_or_body_only_window'}
 if p:
  event=dict(protocol='shoulder_press',laterality='bilateral',posture='seated',closure='observed_rack',rack_visible_stable=True,peak_frames=[round(v*r['fps']) for v in p],turn_frames=[round(v*r['fps']) for v in t]);lo=lower(ss,event,r['fps'],r['width'],r['height'],'front',nf)
 rows.append(dict(r,id=f'{sid}-review-{n+1}',source_id=sid,window_seconds=[a,b],body_grade=g,lower_grade=2 if p else None,cache_path=str(path),cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cache_fingerprint=m['fingerprint'],body=body,lower=lo,event=event,visual_tracking_review='pending',label_origin='AI_visual_review_draft_not_expert_adjudication'))
(O/'feature-candidates.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n');print([(r['id'],r['body']['status'],r['lower']['status'],r['lower'].get('reason')) for r in rows])
