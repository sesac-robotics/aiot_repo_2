import json,math,subprocess,sys
from pathlib import Path
import cv2,numpy as np
B=Path(__file__).resolve().parents[2];D=B/'output/squat-side-return-v3-20261024/mining';O=B/'data/squat-side-return-expansion-20261024';(O/'clips/squats').mkdir(parents=True,exist_ok=True);(O/'annotations').mkdir(exist_ok=True)
sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
labels=json.loads((D/'labels-v2.frozen.json').read_text());manifest=[]
for r in labels['reviewed']:
    start=math.ceil(r['start']*r['fps']-1e-7);end=math.ceil(r['end']*r['fps']-1e-7);target=O/'clips/squats'/(r['id']+'.mkv');assert sha(r['path'])==r['source_sha256']
    assert not target.exists()
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-i',r['path'],'-map','0:v:0','-vf',f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS','-an','-c:v','ffv1','-threads','1','-level','3',str(target)],check=True)
    a=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);c=cv2.VideoCapture(str(target),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);n=int(c.get(7));assert n==end-start;checks=[]
    for f in [0,n//2,n-1]:
        a.set(1,start+f);ok,x=a.read();assert ok;c.set(1,f);ok,y=c.read();assert ok and np.array_equal(x,y);checks.append(dict(clip_frame=f,source_frame=start+f,exact_decoded_pixels_equal=True))
    a.release();c.release();manifest.append(dict(id=r['id'],path=str(target),sha256=sha(target),source_path=r['path'],source_sha256=r['source_sha256'],source_start_frame=start,source_end_frame_exclusive=end,frames=n,fps=r['fps'],split=r['split'],person=r['person'],grades=r['grades'],pixel_checks=checks))
(O/'annotations/segments.frozen.json').write_bytes((D/'labels-v2.frozen.json').read_bytes());(O/'clip-manifest.json').write_text(json.dumps(dict(clips=manifest),indent=2));print('EXPORTED',len(manifest),'pixel checks',sum(len(c['pixel_checks']) for c in manifest))
