import sys,json
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/pushup-root-cause-20261024';sys.path.insert(0,str(B/'five-exercise-comparison'));from crop_extract import extract_view
rows=json.loads((O/'crop-plan-v2.frozen.json').read_text());receipts=[]
for r in rows:
 if r['split']=='test' and '--include-test' not in sys.argv:continue
 rec=extract_view(r,'primary',r['source'],O/'observations');receipts.append(dict(id=r['id'],extraction=rec));(O/('crop-extraction-test.json' if '--include-test' in sys.argv else 'crop-extraction.json')).write_text(json.dumps(receipts,indent=2));print(r['id'],rec['frames'],flush=True)
print('ROI_EXTRACTED',len(receipts),flush=True)
