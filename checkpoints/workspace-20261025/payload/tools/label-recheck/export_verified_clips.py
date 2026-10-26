"""Materialize reviewed intervals; decode every derivative frame and retain provenance."""
from pathlib import Path
import json,subprocess,hashlib,cv2
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/label-verified-five-expanded-20261024';labels=json.loads((O/'new-labels.frozen.json').read_text());inventory={r['id']:r for r in json.loads((B/'output/r2d2-review-20261024/inventory.json').read_text())};D=B/'data/label-recheck-20261024/clips/ytw';D.mkdir(parents=True,exist_ok=True)
segments=[dict(r,destination=str(D/(r['id']+'.mp4'))) for r in labels];team=next(r for r in json.loads((B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json').read_text())['labels'] if r['id']=='V38_014000_025750');segments.append(dict(id=team['id'],source=inventory['V38']['copy'],sources=[team['source_sha256']],start=team['start'],end=team['end'],destination=str(B/team['clip'])))
receipts=[];hashes={}
for r in segments:
 source=Path(r['source']);digest=hashes.setdefault(str(source),hashlib.file_digest(source.open('rb'),'sha256').hexdigest());assert digest==r['sources'][0];p=Path(r['destination']);p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():subprocess.run(['ffmpeg','-v','error','-threads','1','-ss',str(r['start']),'-i',str(source),'-t',str(r['end']-r['start']),'-an','-c:v','libx264','-threads','1','-preset','fast','-crf','20','-movflags','+faststart',str(p)],check=True)
 c=cv2.VideoCapture(str(p),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);fps=c.get(5);n=0
 while c.read()[0]:n+=1
 expected=int(c.get(7));c.release();assert n==expected and n>0;duration=n/fps;assert abs(duration-(r['end']-r['start']))<.13
 receipts.append(dict(id=r['id'],clip=str(p),source=str(source),source_sha256=digest,clip_sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),start=r['start'],end=r['end'],decoded_frames=n,duration=duration));(O/'clip-verification.json').write_text(json.dumps(receipts,indent=2));print(r['id'],n,flush=True)
for source,digest in hashes.items():assert hashlib.file_digest(Path(source).open('rb'),'sha256').hexdigest()==digest
print('CLIPS_VERIFIED',len(receipts),flush=True)
