from pathlib import Path
import sys,json,hashlib
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');sys.path.insert(0,str(B/'output/live-webcam-20261023'))
from shoulder_candidate import ShoulderTracker
from phase_features_v3 import extract,angle
from observation_features_v4 import extract_return
O=B/'output/shoulder-work-20261024/observation-return-v4-20261025/saved-context-v1';O.mkdir(exist_ok=True)
p=B/'output/live-webcam-20261023/sessions/1792849624147364792/pose.npz';z=np.load(p);ticks=z['frame_numbers'].astype(int);x=z['image_xy'];w,h=int(z['width']),int(z['height']);tr=ShoulderTracker();rows=[]
for t,q in zip(ticks,x):
 seg=tr.push(int(t),q,width=w,height=h)
 if not seg:continue
 st,en=seg[0][0],seg[-1][0];xx=np.array([q for _,q in seg]);torso=np.linalg.norm(xx[:,[11,12]].mean(1)-xx[:,[23,24]].mean(1),axis=1)
 el=np.stack([angle(xx,11,13,15),angle(xx,12,14,16)],1);elev=np.stack([angle(xx,23,11,13),angle(xx,24,12,14)],1);wrist=(xx[:,[11,12],1]-xx[:,[15,16],1])/torso[:,None]
 a=dict(number=tr.count,segment=[st,en],ready_elbow=np.median(el[:6],0).tolist(),ready_elevation=np.median(elev[:6],0).tolist(),ready_wrist=np.median(wrist[:6],0).tolist(),end_elbow=np.median(el[-3:],0).tolist(),end_elevation=np.median(elev[-3:],0).tolist(),end_wrist=np.median(wrist[-3:],0).tolist(),contexts=[])
 for extra in [0,3,6,10,15,20]:
  ss=[(int(t),q) for t,q in zip(ticks,x) if st<=t<=en+extra];old=extract(ss,w,h);new=extract_return(ss,w,h)
  a['contexts'].append(dict(extra_slots=extra,supplied_end=ss[-1][0],phase_status=old['status'],phase_reason=old.get('reason'),return_ticks=old.get('return_ticks'),closure=old.get('closure'),top_tick=old.get('top_tick'),new_status=new['status'],new_reason=new.get('reason')))
 rows.append(a)
result=dict(session_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='same saved diagnostic no video ground truth; suffix timing is development analysis only',camera_opened=False,training=False,rows=rows)
(O/'diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for r in rows:
 print(r['number'],r['segment'],'elev_delta',np.array(r['end_elevation'])-r['ready_elevation'],'wrist_delta',np.array(r['end_wrist'])-r['ready_wrist'])
 print([(c['extra_slots'],c['return_ticks'],c['closure'],c['new_status'],c['new_reason']) for c in r['contexts']])
