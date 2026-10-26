import json,sys
from pathlib import Path
B=Path(__file__).resolve().parents[2];O=B/'output/squat-side-return-v3-20261024'
sys.path.insert(0,str(B/'five-exercise-comparison'))
import extract
seen=set()
for r in json.loads((O/'mining/labels.frozen.json').read_text())['reviewed']:
    if r['review_id'] in seen:continue
    seen.add(r['review_id'])
    request=dict(id=r['review_id'],video_sha256=r['source_sha256'],required_views=['front'],segment=dict(start_frame=0,end_frame=r['frame_count']-2,fps=r['fps']))
    rec=extract.extract_view(request,'primary',r['path'],O/'observations')
    print('EXTRACTED',r['review_id'],rec['frames'],flush=True)
