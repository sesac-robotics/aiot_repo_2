"""학습·검증 구간의 실제 정면만 CPU Heavy10Hz로 다시 추출한다."""
import sys,json
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/squat-relabel-20261023';L=B/'output/live-webcam-20261023';sys.path.insert(0,str(L));from runtime_snapshot import prepare_snapshot
runtime,_=prepare_snapshot();sys.path.insert(0,str(runtime));from app.scoring.media import extract_video,save_landmarks
rows=json.loads((O/'source-manifest.json').read_text());extractions=[]
for i,r in enumerate(rows):
 assert r['split'] in ('train','validation')
 p=O/'observations'/r['id'];p.mkdir(parents=True,exist_ok=True)
 seq,meta=extract_video(Path(r['views']['front']['path']),B/'mediapipe-pilot/models/heavy.task',start_frame=r['first_frame'],end_frame=r['last_frame'],delegate='CPU')
 save_landmarks(p/'front.npz',seq);(p/'extraction.json').write_text(json.dumps(meta,indent=2));extractions.append(dict(id=r['id'],path=str(p/'front.npz'),frames=len(seq.frame_numbers)));print(i+1,r['id'],len(seq.frame_numbers),flush=True)
(O/'extraction.json').write_text(json.dumps(extractions,indent=2))
