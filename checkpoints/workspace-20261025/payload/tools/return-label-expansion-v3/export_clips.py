import json,sys,math,subprocess
from pathlib import Path
import numpy as np,cv2
B=Path(__file__).resolve().parents[2];O=B/'data/return-label-expansion-v3-20261024';O.mkdir(exist_ok=True);D=B/'output/return-label-expansion-v3-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import assert_training_allowed
labels=json.loads((D/'labels-v2.frozen.json').read_text())['reviewed'];assert_training_allowed(labels);manifest=[]
for r in labels:
 core_start=math.ceil(r['start']*r['fps']);core_end=math.ceil(r['end']*r['fps']);first=max(0,core_start-round(.3*r['fps']));end=min(r['frame_count'],core_end+round(.3*r['fps']));p=O/(r['id']+'.mkv')
 if not p.exists():subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-i',r['path'],'-map','0:v:0','-vf',f'trim=start_frame={first}:end_frame={end},setpts=PTS-STARTPTS','-an','-c:v','ffv1','-threads','1',str(p)],check=True)
 a=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);b=cv2.VideoCapture(str(p),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);n=int(b.get(7));assert n==end-first;checks=[]
 for f in [0,n//2,n-1]:
  a.set(1,first+f);ok,x=a.read();assert ok;b.set(1,f);ok,y=b.read();assert ok and np.array_equal(x,y);checks.append(dict(source_frame=first+f,clip_frame=f,exact_decoded_pixels_equal=True))
 a.release();b.release();manifest.append(dict(id=r['id'],person=r['person'],split=r['split'],grade=r['return_grade'],criterion='return_extension',path=str(p),sha256=sha(p),original=r['path'],source_sha256=sha(r['path']),source_frame_range=[first,end],labeled_core_frame_range=[core_start,core_end],extra_context_is_not_a_separate_label=True,fps=r['fps'],frames=n,pixel_checks=checks))
assert len(manifest)==3 and sum(len(r['pixel_checks']) for r in manifest)==9
p=O/'clip-manifest-v2.frozen.json';assert not p.exists();p.write_text(json.dumps(dict(labels_sha256=sha(D/'labels-v2.frozen.json'),clips=manifest,lossless=True,Train_only=True,borderline_held_clip_not_training=True,supersedes_incomplete_manifest='clip-manifest.frozen.json'),indent=2));print('Exported3 clips,9 exact pixel checks')
