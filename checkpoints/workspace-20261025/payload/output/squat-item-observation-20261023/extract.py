import json,sys
from pathlib import Path
O=Path(__file__).resolve().parent;B=O.parents[1];sys.path.insert(0,str(B/'output/live-webcam-20261023'))
from runtime_snapshot import prepare_snapshot
runtime,_=prepare_snapshot();sys.path.insert(0,str(runtime));from app.scoring.media import extract_video,save_landmarks
rows=json.loads((O/'review-manifest.json').read_text())
for i,r in enumerate(rows):
 p=O/'observations'/r['id'];p.mkdir(parents=True,exist_ok=True)
 if (p/'front.npz').exists():continue
 seq,meta=extract_video(Path(r['path']),B/'mediapipe-pilot/models/heavy.task',delegate='CPU',start_frame=0,end_frame=r['frame_count']-1)
 save_landmarks(p/'front.npz',seq);(p/'extraction.json').write_text(json.dumps(meta,indent=2));print(i+1,r['id'],len(seq.frame_numbers),flush=True)
