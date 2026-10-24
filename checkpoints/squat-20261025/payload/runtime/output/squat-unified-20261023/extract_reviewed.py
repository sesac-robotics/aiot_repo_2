from pathlib import Path
import sys,json
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent;sys.path.insert(0,str(R/'output/live-webcam-20261023'))
from runtime_snapshot import prepare_snapshot
runtime,_=prepare_snapshot();sys.path.insert(0,str(runtime));from app.scoring.media import extract_video,save_landmarks
for r in json.loads((O/'new_visual_labels.json').read_text())['records']:
 p=O/'observations'/r['id'];p.mkdir(exist_ok=True,parents=True)
 if (p/(r['view']+'.npz')).exists():continue
 seq,meta=extract_video(Path(r['source_path']),R/'mediapipe-pilot/models/heavy.task',start_frame=r['start_frame'],end_frame=r['end_frame'],delegate='CPU');save_landmarks(p/(r['view']+'.npz'),seq);(p/'extraction.json').write_text(json.dumps(meta,indent=2));print(r['id'],len(seq.frame_numbers),flush=True)
