from pathlib import Path
import json,sys,hashlib
B=Path('/home/jayden/SeSACPAI-Mid');sys.path.insert(0,str(B/'five-exercise-comparison'));import extract
O=B/'output/shoulder-work-20261024/grade-support-v1-20261025'
inv={r['id']:r for r in json.loads((B/'output/shoulder-work-20261024/full-review-v1/inventory.frozen.json').read_text())['sources']};out=[]
for sid in ['subject_012_shoulder_bad_front','subject_012_shoulder_good_front']:
 s=inv[sid];assert s['split']=='validation';assert hashlib.sha256(Path(s['path']).read_bytes()).hexdigest()==s['source_sha256']
 row=dict(id=sid,video_sha256=s['source_sha256'],front_sha256=s['source_sha256'],required_views=['front'],segment=dict(start_frame=0,end_frame=48,fps=15.))
 r=extract.extract_view(row,'primary',s['path'],O/'observations');out.append(dict(id=sid,split=s['split'],source_sha256=s['source_sha256'],extraction=r))
 print(sid,r['frames'],flush=True)
p=O/'validation-extraction.frozen.json';text=json.dumps(out,ensure_ascii=False,indent=2)+'\n'
if p.exists():assert p.read_text()==text
else:p.write_text(text)
