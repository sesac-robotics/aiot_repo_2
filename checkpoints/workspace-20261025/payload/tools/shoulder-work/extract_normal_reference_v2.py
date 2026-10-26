from pathlib import Path
import json,sys,os
B=Path('/home/jayden/SeSACPAI-Mid');sys.path.insert(0,str(B/'five-exercise-comparison'));import extract
O=B/'output/shoulder-work-20261024/normal-reference-v2'
rows=json.loads((O/'plan.frozen.json').read_text())['rows'];receipts=[]
for s in rows:
 row=dict(id=s['id'],video_sha256=s['source_sha256'],front_sha256=s['source_sha256'],required_views=[s['view']],segment=dict(start_frame=round(s['extract_start']*15),end_frame=round(s['extract_end']*15),fps=15.))
 r=extract.extract_view(row,'primary',s['path'],O/'observations');receipts.append(dict(id=s['id'],extraction=r));(O/'extraction-progress.json').write_text(json.dumps(receipts,indent=2));print(s['id'],r['frames'],flush=True)
p=O/'extraction.frozen.json';assert not p.exists();p.write_text(json.dumps(receipts,indent=2))
