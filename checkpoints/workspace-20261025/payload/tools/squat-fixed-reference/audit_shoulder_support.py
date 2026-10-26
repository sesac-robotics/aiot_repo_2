"""Audit common shoulder data without fitting or changing exercise-owned candidates."""
import collections, datetime, json, sys
from pathlib import Path
import cv2
B=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import POLICY, assert_training_allowed
from diagnose_remaining import sheet
P=B/'output/squat-fixed-reference-combined-v3-20261024'
O=B/'output/shoulder-common-support-audit-20261024'
read=lambda p:json.loads(p.read_text())
def save(n,x):
 p=O/n;assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def main():
 O.mkdir(exist_ok=True);rows=read(P/'score-inputs.frozen.json');reports=read(P/'score-training.json')['heads']
 counts=collections.Counter();errors=[]
 for r in rows:
  if r['exercise']!='shoulder':continue
  for i,g in zip(r['items'],r['levels']):
   if i['item'] not in ['lower_return','frontal_trunk_hold']:continue
   counts[i['item'],i['bank'],r['family'],r['split'],g,i['eligible']]+=1
   if g is not None and g<2:errors.append(dict(id=r['id'],person=r['person'],split=r['split'],sources=r['sources'],item=i['item'],grade=g,eligible=i['eligible'],holds=i['holds']))
 shared=B/'output/per-person-stratified-80-20-20261024/shared-trunk-training.json'
 inputs=B/'output/per-person-stratified-80-20-20261024/shared-trunk-inputs.frozen.json'
 save('support-audit.frozen.json',dict(parent=str(P),parent_inputs_sha256=sha(P/'score-inputs.frozen.json'),
   policy_sha256=sha(POLICY),cells=[dict(item=k[0],bank=k[1],family=k[2],split=k[3],grade=k[4],eligible=k[5],count=v) for k,v in counts.items()],
   item_reports=[r for r in reports if r['key'][0]=='shoulder' and r['key'][1] in ['lower_return','frontal_trunk_hold']],
   annotated_errors=errors,shared_trunk_report_path=str(shared),shared_trunk_sha256=sha(shared),
   shared_trunk_error_inputs=[r for r in read(inputs) if r['target']<2],
   no_fit=True,production_changes=False,interpretation='Shared trunk errors originate in squats, not shoulder. No direct eligible shoulder errors currently support these heads.'))
 pool=read(B/'output/overnight-20261020/score_candidates.json');lookup={r['id']:r for r in pool}
 roles={r['person']:r['split'] for r in rows if r['family']!='R2D2'}
 candidates=[]
 for review_id,num,seconds in [('T13','013',[23,29]),('T16','016',[14,19]),('T19','019',None),('T23','023',None),('T11','011',None)]:
  rid=f'subject_{num}_shoulder_bad_front';r=lookup[rid];assert roles.get(r['person'],r['split'])=='train'
  cap=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);fps=cap.get(cv2.CAP_PROP_FPS);N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
  start,end=seconds or [0,N/fps]
  frames=list(range(round(start*fps),min(N,round(end*fps)),max(1,round(fps/5))))
  # For longer originals record a coarse whole-video screen; targeted windows stay5Hz.
  if len(frames)>64:frames=list(range(0,N,max(1,round(fps*1.0))))
  candidate=dict(id=rid,review_id=review_id,path=r['path'],person=r['person'],split='train',source_sha256=sha(r['path']),sources=[sha(r['path'])],review_frames=frames,fps=fps,source_role_preserved=True)
  assert_training_allowed([candidate]);candidates.append(candidate)
 inv={r['id']:r for r in read(B/'output/r2d2-review-20261024/inventory.json')};policy=read(POLICY)
 assert policy['assignments']['V27']=='train';r=inv['V27'];fps=29.9984623967
 frames=list(range(round(16.9*fps),round(20.3*fps),3))
 candidates.append(dict(id='V27_trunk_recheck',review_id='TEAM27',path=r['copy'],person='P04',split='train',
   source_sha256=r['sha256'],sources=[r['sha256']],review_frames=frames,fps=fps,source_role_preserved=True))
 assert_training_allowed(candidates)
 save('review-intake.frozen.json',dict(frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   target='frontal_trunk_stability_only',source_roles_preserved=True,not_using_Test=True,labels_not_assigned=True,records=candidates))
 for r in candidates:
  assert sha(r['path'])==r['source_sha256']
  for page,start in enumerate(range(0,len(r['review_frames']),32),1):
   dest=O/(r['review_id']+f'-{page:02}.jpg');assert not dest.exists()
   sheet(r['path'],r['review_frames'][start:start+32],dest,r['review_id']+' Train original; frontal trunk target only')
  print('SAVED',r['review_id'],len(r['review_frames']),flush=True)
if __name__=='__main__':main()
