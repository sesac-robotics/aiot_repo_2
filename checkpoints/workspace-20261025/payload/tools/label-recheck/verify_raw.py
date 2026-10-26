import json,hashlib
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/label-verified-five-expanded-20261024';rows=json.loads((B/'output/r2d2-review-20261024/inventory.json').read_text());receipts=[]
for r in rows:
 p=Path(r['copy']);digest=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();assert digest==r['sha256'];original=Path(r['source']);original_digest=hashlib.file_digest(original.open('rb'),'sha256').hexdigest() if original.exists() else None;assert original_digest is None or original_digest==digest;receipts.append(dict(source_id=r['id'],copy=str(p),copy_sha256=digest,download_source_exists=original.exists(),download_source_sha256=original_digest))
(O/'raw-verification.json').write_text(json.dumps(receipts,indent=2));print('RAW_VERIFIED',len(receipts),flush=True)
