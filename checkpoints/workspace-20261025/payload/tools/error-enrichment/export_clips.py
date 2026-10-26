from pathlib import Path
import json,subprocess,hashlib,cv2
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/error-enrichment-20261024';D=B/'data/error-enrichment-20261024';rows=json.loads((D/'annotations/segments.frozen.json').read_text());out=[]
rows += json.loads((D/'annotations/rehab-segments.frozen.json').read_text())
for r in rows:
 p=D/'clips'/r['exercise']/(r['id']+'.mp4');p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():subprocess.run(['ffmpeg','-v','error','-threads','1','-ss',str(r['start']),'-i',r['path'],'-t',str(r['end']-r['start']),'-an','-map_metadata','-1','-c:v','libx264','-preset','fast','-crf','20','-threads','1','-movflags','+faststart',str(p)],check=True,timeout=120)
 c=cv2.VideoCapture(str(p),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);count=0;fps=c.get(5)
 while c.grab():count+=1
 c.release();assert count>=2 and abs(count/fps-(r['end']-r['start']))<.12
 out.append(dict(id=r['id'],clip=str(p),sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),source=r['path'],source_sha256=r['source_sha256'],start=r['start'],end=r['end'],frames_decoded=count,fps=fps,source_unchanged=hashlib.file_digest(Path(r['path']).open('rb'),'sha256').hexdigest()==r['source_sha256'],training_eligible=r['train_eligible'],audio_in_derivative=False));(D/'clip-manifest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(r['id'],count,flush=True)
print('CLIPS_COMPLETE',len(out),flush=True)
