import json,sys
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/ai-rule-paired-20261024';sys.path.insert(0,str(B/'five-exercise-comparison'));import extract
rows=json.loads((O/'reviewed-intake-v2.frozen.json').read_text())['labels'];receipt=[]
for r in rows:
 if not r['train_eligible']:continue
 p=dict(id=r['review_id'],video_sha256=r['source_sha256'],required_views=['front'],segment=dict(start_frame=0,end_frame=r['frame_count']-2,fps=r['fps']))
 q=extract.extract_view(p,'primary',r['path'],O/'observations-v2');receipt.append(dict(id=r['review_id'],extraction=q));(O/'new-extraction.json').write_text(json.dumps(receipt,indent=2));print(r['review_id'],q['frames'],flush=True)
print('EXTRACTION_COMPLETE',len(receipt),flush=True)
