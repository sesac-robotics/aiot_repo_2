"""Export reviewed frame-bounded cycles losslessly with original-source lineage."""
import json,subprocess,sys,math,shutil
from pathlib import Path
import cv2,numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');D=B/'output/return-failure-expansion-20261024';O=B/'data/return-failure-expansion-20261024';(O/'annotations').mkdir(parents=True,exist_ok=True);(O/'clips/pushups').mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
labels=json.loads((D/'labels-v2.frozen.json').read_text());manifest=[]
for r in labels['reviewed']:
 start=math.ceil(r['start']*r['fps']-1e-7);end=math.ceil(r['end']*r['fps']-1e-7);path=O/'clips/pushups'/(r['id']+'.mkv');assert sha(r['path'])==r['source_sha256']
 if not path.exists():subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-i',r['path'],'-map','0:v:0','-vf',f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS','-an','-c:v','ffv1','-threads','1','-level','3',str(path)],check=True)
 a=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);b=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);count=int(b.get(7));assert count==end-start;checks=[]
 for frame in [0,count//2,count-1]:
  a.set(1,start+frame);ok,x=a.read();assert ok;b.set(1,frame);ok,y=b.read();assert ok and np.array_equal(x,y);checks.append(dict(clip_frame=frame,source_frame=start+frame,exact_decoded_pixels_equal=True))
 a.release();b.release();manifest.append(dict(id=r['id'],path=str(path),sha256=sha(path),source_path=r['path'],source_sha256=r['source_sha256'],source_start_frame=start,source_end_frame_exclusive=end,frames=count,fps=r['fps'],person=r['person'],group=r['person'],split=r['split'],criterion='return_extension',grade=r['grades'][2],other_criteria_unlabeled=True,frame_mapping='source_frame=clip_frame+source_start_frame',pixel_checks=checks));print('CLIP_VERIFIED',r['id'],count,flush=True)
dest=O/'annotations/segments.frozen.json'
if dest.exists():assert sha(dest)==sha(D/'labels-v2.frozen.json')
else:shutil.copyfile(D/'labels-v2.frozen.json',dest)
(O/'clip-manifest.json').write_text(json.dumps(dict(source_role_preserved=True,exact_frame_exports=True,clips=manifest),indent=2));print('EXPORTED',len(manifest))
