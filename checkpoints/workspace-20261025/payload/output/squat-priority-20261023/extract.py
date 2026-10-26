from pathlib import Path
import json,sys,hashlib
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent;sys.path.insert(0,str(R/'output/live-webcam-20261023'))
from runtime_snapshot import prepare_snapshot
runtime,_=prepare_snapshot();sys.path.insert(0,str(runtime));from app.scoring.media import extract_video,save_landmarks
raw=(O/'labels.frozen.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==(O/'labels.sha256').read_text()
for j,r in enumerate(json.loads(raw)['labels']):
 p=O/'observations'/r['id'];p.mkdir(exist_ok=True,parents=True)
 if (p/'front.npz').exists():continue
 seq,meta=extract_video(Path(r['path']),R/'mediapipe-pilot/models/heavy.task',start_frame=r['frames'][0],end_frame=r['frames'][1],delegate='CPU');save_landmarks(p/'front.npz',seq);(p/'extraction.json').write_text(json.dumps(meta,indent=2));print(j+1,r['id'],len(seq.frame_numbers),flush=True)
