import json,sys
from pathlib import Path
O=Path(__file__).resolve().parent;B=O.parents[1];sys.path.insert(0,str(B/'output/live-webcam-20261023'));from runtime_snapshot import prepare_snapshot
rt,_=prepare_snapshot();sys.path.insert(0,str(rt));from app.scoring.media import extract_video,save_landmarks
for i,r in enumerate(json.loads((O/'manifest.json').read_text())):
 p=O/'observations'/r['id'];p.mkdir(parents=True,exist_ok=True)
 if (p/'front.npz').exists():continue
 seq,meta=extract_video(Path(r['path']),B/'mediapipe-pilot/models/heavy.task',start_frame=r['first_frame'],end_frame=r['last_frame'],delegate='CPU');save_landmarks(p/'front.npz',seq);print(i+1,r['id'],len(seq.frame_numbers),flush=True)
