from pathlib import Path
import sys,json,hashlib
import numpy as np
O=Path(__file__).resolve().parent;B=O.parents[1];A=B/'output/live-webcam-20261023';sys.path.insert(0,str(A));import server
from movement import MovementTracker
sid='1792744784509293036';z=np.load(A/'sessions'/sid/'pose.npz');tracker=MovementTracker('squats','front');rows=[]
for t,x in zip(z['frame_numbers'],z['image_xy']):
 seg=tracker.push(int(t),x)
 if seg:
  return_samples=server.finish_observation_window(seg,tracker.standing_reference,[(int(t),x) for t,x in zip(z['frame_numbers'],z['image_xy'])])
  r=server.score_samples(seg,int(z['width']),int(z['height']),'squats','front',standing_reference=tracker.standing_reference,return_samples=return_samples)
  rows.append(dict(number=tracker.count,duration=round((seg[-1][0]-seg[0][0])/10,1),start_tick=seg[0][0],end_tick=seg[-1][0],**{k:r.get(k) for k in ['ai','rule','items','front_candidate','coaching','accepted_items','method_comparison','score_100','legacy_score_100']}));print(tracker.count,r['ai'],flush=True)
d=dict(collecting=False,exercise='squats',view='front',status='저장 관절 자료 재평가 · 새 촬영 아님',movement=tracker.status(),repetitions=rows,last_rep=rows[-1],coaching=rows[-1]['coaching'],source_session=sid,replay_only=True,standing_state_version='initial-standing-state-v1',source_pose_sha256=hashlib.sha256((A/'sessions'/sid/'pose.npz').read_bytes()).hexdigest());(A/'standing-state/replay-state.json').write_text(json.dumps(server.safe(d),ensure_ascii=False,indent=2));(O/'replay-state.json').write_text(json.dumps(server.safe(d),ensure_ascii=False,indent=2))
