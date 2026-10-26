from pathlib import Path
import sys,json
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/error-enrichment-20261024';sys.path.insert(0,str(B/'five-exercise-comparison'));import extract
rows=json.loads((B/'data/error-enrichment-20261024/annotations/rehab-segments.frozen.json').read_text());receipts=json.loads((O/'extraction.json').read_text());known={r['review_id'] for r in receipts}
for r in rows:
 if r['review_id'] in known:continue
 row=dict(id=r['review_id'],video_sha256=r['source_sha256'],required_views=['front'],segment=dict(start_frame=r['first_frame'],end_frame=r['last_frame'],fps=r['fps']));rec=extract.extract_view(row,'primary',r['path'],O/'observations');receipts.append(dict(review_id=r['review_id'],source=r['path'],extraction=rec));(O/'extraction.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2));print(r['review_id'],rec['frames'],flush=True)
print('REHAB_EXTRACTION_COMPLETE',flush=True)
