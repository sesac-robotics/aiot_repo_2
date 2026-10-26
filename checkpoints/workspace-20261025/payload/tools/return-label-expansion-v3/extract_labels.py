import json,sys
from pathlib import Path
B=Path(__file__).resolve().parents[2];O=B/'output/return-label-expansion-v3-20261024';sys.path.insert(0,str(B/'five-exercise-comparison'));import extract
labels=json.loads((O/'labels.frozen.json').read_text())['reviewed'];seen=set();receipts=[]
for r in labels:
 if r['review_id'] in seen:continue
 seen.add(r['review_id']);row=dict(id=r['review_id'],video_sha256=r['source_sha256'],required_views=['front'],segment=dict(start_frame=0,end_frame=r['frame_count']-2,fps=r['fps']))
 rec=extract.extract_view(row,'primary',r['path'],O/'observations',delegate='CPU');receipts.append(dict(review_id=r['review_id'],extraction=rec));(O/'extraction.json').write_text(json.dumps(receipts,indent=2));print('EXTRACTED',r['review_id'],rec['frames'],flush=True)
