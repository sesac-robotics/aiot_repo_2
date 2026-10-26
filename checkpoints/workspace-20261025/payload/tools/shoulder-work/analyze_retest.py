from pathlib import Path
import sys,json,collections,hashlib
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');L=B/'output/live-webcam-20261023';sys.path.insert(0,str(L))
from shoulder_candidate import ShoulderTracker
from movement import joint_angle
from live_observation import mask_frame_border
S=L/'sessions/1792848383011095301';p=np.load(S/'pose.npz');d=json.loads((S/'final-state.json').read_text());tr=ShoulderTracker();rows=[]
for tick,raw in zip(p['frame_numbers'],p['image_xy']):
 x=mask_frame_border(raw,int(p['width']),int(p['height']));tr.push(int(tick),raw,int(p['width']),int(p['height']))
 req=[11,12,13,14,15,16,23,24];valid=bool(np.isfinite(x[req]).all());r={'tick':int(tick),'phase':tr.phase,'ready':tr.ready,'active':tr.active,'has_top':tr.has_top,'missing':[j for j in req if not np.isfinite(x[j]).all()],'low':False,'high':False}
 if valid:
  a=[joint_angle(x,*j) for j in [(11,13,15),(12,14,16)]];scale=max(30.,float(np.linalg.norm(x[[11,12]].mean(0)-x[[23,24]].mean(0))));w=(x[[11,12],1]-x[[15,16],1])/scale;e=(x[[11,12],1]-x[[13,14],1])/scale
  if all(z is not None for z in a):r.update(angles=a,wrist_lift=w.tolist(),elbow_lift=e.tolist(),low=bool(max(a)<=110 and max(e)<.2),high=bool(min(a)>=150 and min(w)>=.35 and min(e)>=.05))
 rows.append(r)
def runs(key):
 out=[];start=None;last=None
 for r in rows:
  if r[key]:
   if start is None or r['tick']-last>1:
    if start is not None:out.append([start,last,last-start+1])
    start=r['tick']
   last=r['tick']
  elif start is not None:out.append([start,last,last-start+1]);start=None
 if start is not None:out.append([start,last,last-start+1])
 return out
summary={'session':str(S),'pose_sha256':hashlib.sha256((S/'pose.npz').read_bytes()).hexdigest(),'samples':len(rows),'duration_seconds':(rows[-1]['tick']-rows[0]['tick'])/10,'saved_count':d['movement']['count'],'replay_count':tr.count,'phase_counts':dict(collections.Counter(r['phase'] for r in rows)),'missing_joint_counts':dict(collections.Counter(j for r in rows for j in r['missing'])),'low_samples':sum(r['low'] for r in rows),'high_samples':sum(r['high'] for r in rows),'max_ready':max(r['ready'] for r in rows),'active_samples':sum(r['active'] for r in rows),'top_samples':sum(r['has_top'] for r in rows),'low_runs':runs('low'),'high_runs':runs('high'),'tick_deltas':dict(collections.Counter(np.diff(p['frame_numbers']).tolist()))}
O=B/'output/shoulder-work-20261024/retest-1792848383011095301';O.mkdir(exist_ok=True);(O/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(O/'frames.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
