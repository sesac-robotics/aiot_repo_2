"""Extract only reviewed source windows, with source/model fingerprints."""
import sys,json,cv2
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/error-enrichment-20261024';sys.path.insert(0,str(B/'five-exercise-comparison'));import extract
rows=json.loads((O/'intake.json').read_text());D=B/'data/error-enrichment-20261024/annotations';labels=json.loads((D/'segments.frozen.json').read_text());states=json.loads((D/'standing-states.frozen.json').read_text());needed={r['review_id'] for r in labels if r['train_eligible']}|{r['review_id'] for r in states};receipts=[]
for r in rows:
 if r['review_id'] not in needed:continue
 a,b=(6.5,15.25) if r['exercise']=='shoulder' else (0,(r['frame_count']-2)/r['fps'])
 row=dict(id=r['review_id'],video_sha256=r['sha256'],required_views=['front'],segment=dict(start_frame=round(a*r['fps']),end_frame=min(r['frame_count']-2,round(b*r['fps'])),fps=r['fps']))
 rec=extract.extract_view(row,'primary',r['path'],O/'observations');receipts.append(dict(review_id=r['review_id'],source=r['path'],extraction=rec));(O/'extraction.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2));print(r['review_id'],rec['frames'],flush=True)
print('EXTRACTION_COMPLETE',len(receipts),flush=True)
