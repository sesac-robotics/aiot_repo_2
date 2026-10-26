from pathlib import Path
import numpy as np,json,sys
from movement_candidate import MovementTracker
from standing_state import evaluate_return
O=Path(__file__).resolve().parent;B=O.parents[1];sys.path.insert(0,str(B/'output/live-webcam-20261023'));results=[]
for sid in ['1792740109492452117','1792744784509293036']:
 z=np.load(B/'output/live-webcam-20261023/sessions'/sid/'pose.npz');tr=MovementTracker('squats','front');rows=[]
 for t,p in zip(z['frame_numbers'],z['image_xy']):
  seg=tr.push(int(t),p)
  if seg:
   from server import finish_observation_window
   seg=finish_observation_window(seg,tr.standing_reference,[(int(t),x) for t,x in zip(z['frame_numbers'],z['image_xy'])])
   rows.append(dict(number=tr.count,score=evaluate_return(np.array([p for _,p in seg]),tr.standing_reference,int(z['width']),int(z['height']))))
 r=dict(session=sid,count=tr.count,rows=rows);results.append(r);print(sid,tr.count,[(r['score'].get('ai_points'),r['score'].get('standing_confirmed')) for r in rows],flush=True)
(O/'session-replay.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
