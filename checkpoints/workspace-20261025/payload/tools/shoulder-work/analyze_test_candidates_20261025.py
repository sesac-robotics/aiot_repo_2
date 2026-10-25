from pathlib import Path
import json,sys,collections,numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');sys.path.insert(0,str(B/'output/live-webcam-20261023'))
from shoulder_candidate import ShoulderTracker
from return_context_v1 import ReturnContextQueue
from observation_features_v4 import extract_return,extract_body
from anatomical_events_v1 import AnatomicalTurnDetector
S=B/'output/live-webcam-20261023/sessions/1792856314599071156';O=B/'output/shoulder-work-20261024/success-1792856314599071156';z=np.load(S/'pose.npz');w,h=int(z['width']),int(z['height']);tr=ShoulderTracker();q=ReturnContextQueue();detector=AnatomicalTurnDetector(w,h);results=[];rows=[];events=[]
for t,xy in zip(z['frame_numbers'],z['image_xy']):
 t=int(t);results.extend(q.push(t,xy));e=detector.push(t,xy,t)
 if e:events.append(e)
 seg=tr.push(t,xy,w,h)
 if seg:
  before=extract_return(seg,w,h);body=extract_body(seg,[seg[0][0],seg[-1][0]],True,w,h)
  rows.append(dict(attempt=tr.count,original_window=[seg[0][0],seg[-1][0]],before=before,body=body))
  r=q.submit(tr.count,seg,w,h)
  if r:results.append(r)
results.extend(q.finish())
for r in results:r.pop('feature_input_samples')
report=dict(scope='new saved session functional candidate replay without video truth; no training',original_attempts=tr.count,original_return_features_ready=sum(r['before']['status']=='features_ready' for r in rows),body_features_ready=sum(r['body']['status']=='features_ready' for r in rows),post_context_features_ready=sum(r['features']['status']=='features_ready' for r in results),post_context_reasons=dict(collections.Counter(r['features'].get('reason','features_ready') for r in results)),anatomical_turn_candidate_events=len(events),anatomical_features_ready=sum(e['features']['status']=='features_ready' for e in events),originals=rows,post_context=results,anatomical_events=events,camera_capture=False,training=False,production_changed=False)
(O/'candidate-replay.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['originals','post_context','anatomical_events']},ensure_ascii=False))
