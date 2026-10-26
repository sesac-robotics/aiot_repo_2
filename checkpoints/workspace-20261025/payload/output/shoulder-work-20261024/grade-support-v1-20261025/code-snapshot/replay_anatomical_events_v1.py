from pathlib import Path
import json,numpy as np
from anatomical_events_v1 import AnatomicalTurnDetector
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-work-20261024/grade-support-v1-20261025';rows=[]
for item in json.loads((O/'validation-feature-diagnostics.json').read_text()):
 z=np.load(item['cache']);dims=z['image_xy'];m=json.loads(Path(item['cache']).with_suffix('.json').read_text())
 import cv2
 inv=json.loads((B/'output/shoulder-work-20261024/full-review-v1/inventory.frozen.json').read_text())['sources'];s=next(r for r in inv if r['id']==item['id']);c=cv2.VideoCapture(s['path']);w,h=int(c.get(3)),int(c.get(4));c.release()
 detector=AnatomicalTurnDetector(w,h,posture='seated',source_fps=15)
 times=z['time_seconds'];ticks=np.round((times-times[0])*10).astype(int)+round(times[0]*10);events=[]
 for t,xy,fr in zip(ticks,z['image_xy'],z['frame_numbers']):
  r=detector.push(int(t),xy,int(fr))
  if r:events.append(r)
 rows.append(dict(id=item['id'],split='validation',events=events,visual_turn_frames=item['annotation']['turn_frames']))
(O/'automatic-event-diagnostics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
for r in rows:print(r['id'],[(e['event']['turn_frames'],e['confirmed_at_tick'],e['features']['status']) for e in r['events']])
