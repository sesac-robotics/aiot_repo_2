import json,sys,subprocess
from pathlib import Path
import cv2,numpy as np
B=Path(__file__).resolve().parents[2];D=B/'data/return-recheck-20261024';(D/'calibration-references').mkdir(exist_ok=True);(D/'review-clips').mkdir(exist_ok=True);sys.path.insert(0,str(B/'tools/error-enrichment'));from common import sha
read=lambda p:json.loads(p.read_text());labels=read(B/'output/squat-return-20261023/labels-v2.frozen.json')['labels'];cases=[]
for p in sorted((B/'output/squat-return-20261023/cfrep').glob('*.mp4')):
 role=next(r['split'] for r in labels if r['path']==str(p));start,end=(369,379) if 'diag_m4' in p.stem else (0,3) if 'front_m4' in p.stem else (0,7);cases.append(dict(id=p.stem+'-ready',path=str(p),start=start,end=end,split=role,kind='calibration-reference',grade=None))
for suffix in ['r05','r08','r11']:
 r=next(r for r in labels if r['id']=='squat_diag_w2_5_7_'+suffix);cases.append(dict(id=r['id']+'-context',path=r['path'],start=r['first_frame'],end=r['last_frame']+13,original_attempt_end_frame=r['last_frame'],split='validation',kind='review-case-context',grade=r['grade']))
manifest=[]
for r in cases:
 out=D/('calibration-references' if r['kind']=='calibration-reference' else 'review-clips')/(r['id']+'.mkv');assert not out.exists();subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-i',r['path'],'-map','0:v:0','-vf',f"trim=start_frame={r['start']}:end_frame={r['end']},setpts=PTS-STARTPTS",'-an','-c:v','ffv1','-threads','1',str(out)],check=True)
 a=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);b=cv2.VideoCapture(str(out),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);n=int(b.get(7));assert n==r['end']-r['start'];checks=[]
 for f in [0,n//2,n-1]:
  a.set(1,r['start']+f);ok,x=a.read();assert ok;b.set(1,f);ok,y=b.read();assert ok and np.array_equal(x,y);checks.append(dict(source_frame=r['start']+f,clip_frame=f,exact_decoded_pixels_equal=True))
 a.release();b.release();manifest.append(dict(r,clip=str(out),clip_sha256=sha(out),source_sha256=sha(r['path']),frames=n,fps=30,training_eligible=r['kind']=='calibration-reference' and r['split']=='train',pixel_checks=checks));print('EXPORTED',r['id'],flush=True)
(D/'clip-manifest.frozen.json').write_text(json.dumps(dict(clips=manifest,validation_never_used_for_fit=True,calibration_clips_not_severity_grade_samples=True),indent=2));print('COMPLETE',len(manifest),'clips',sum(len(m['pixel_checks']) for m in manifest),'pixelchecks')
