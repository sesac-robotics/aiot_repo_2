import json,subprocess,sys
from pathlib import Path
import cv2,numpy as np
B=Path(__file__).resolve().parents[2];O=B/'data/mild-return-expansion-20261024';D=B/'output/mild-return-mining-20261024';(O/'clips/squats').mkdir(parents=True,exist_ok=True);(O/'annotations').mkdir(exist_ok=True);sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
read=lambda p:json.loads(p.read_text());r=next(r for r in read(D/'intake.frozen.json') if r['review_id']=='C28');revisions=read(B/'output/mild-return-fit-20261024/label-revisions.frozen.json');manifest=[]
for kind,start,end,grade in [('attempt',313,352,1),('standing_reference',64,71,None)]:
 p=O/'clips/squats'/(r['id']+'-'+kind+'.mkv');assert not p.exists();assert sha(r['path'])==r['source_sha256']
 subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','1','-i',r['path'],'-map','0:v:0','-vf',f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS','-an','-c:v','ffv1','-threads','1','-level','3',str(p)],check=True)
 a=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);c=cv2.VideoCapture(str(p),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);n=int(c.get(7));assert n==end-start;checks=[]
 for f in [0,n//2,n-1]:
  a.set(1,start+f);ok,x=a.read();assert ok;c.set(1,f);ok,y=c.read();assert ok and np.array_equal(x,y);checks.append(dict(clip_frame=f,source_frame=start+f,exact_decoded_pixels_equal=True))
 a.release();c.release();manifest.append(dict(id=r['id'],kind=kind,path=str(p),sha256=sha(p),source_path=r['path'],source_sha256=r['source_sha256'],source_start_frame=start,source_end_frame_exclusive=end,frames=n,fps=r['fps'],person=r['person'],split='train',criterion='standing_height_return',grade=grade,calibration_only=kind!='attempt',pixel_checks=checks))
(O/'annotations/label-revisions.frozen.json').write_text(json.dumps(revisions,indent=2));(O/'clip-manifest.json').write_text(json.dumps(dict(clips=manifest,source_roles_preserved=True),indent=2))
review=[]
for a in read(D/'intake.frozen.json'):
 review.append(dict(id=a['id'],review_id=a['review_id'],person=a['person'],split=a['split'],old_grade=2,review_decision='correct_to_partial' if a['review_id']=='C28' else 'insufficient_evidence_to_change_existing_grade',new_grade=1 if a['review_id']=='C28' else 2,not_newly_confirmed_normal=a['review_id']!='C28'))
(D/'review-decisions.frozen.json').write_text(json.dumps(dict(reviewed=review,sheets_viewed=[f'review-{i:02}.jpg' for i in range(1,9)],expert_verified=False,no_unobserved_or_stopped_attempts_labeled_as_grade0=True),indent=2));print('EXPORTED2clips;6pixelchecks;32reviewdecisions')
