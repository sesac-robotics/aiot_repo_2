"""Freeze manual sampled-video annotations and extract approved native-Train sources."""
import json,sys,hashlib
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/label-verified-five-expanded-20261024';O.mkdir(exist_ok=True)
spec={1:[(0,3.5,[2,2,0,2]),(3.5,5.9,[2,2,0,2])],2:[(0,2.5,[0,2,2,2]),(2.5,4.5,[0,2,2,2])],3:[(0,2.75,[1,2,None,2]),(2.75,5,[1,2,None,2])],4:[(0,4.5,[0,1,0,2])],5:[(0,2.75,[1,2,None,2]),(3,5.9,[1,2,None,2])],6:[(0,2.9,[1,2,None,None]),(3,5.8,[1,2,None,None])],7:[(3.25,6,[2,2,0,2])],8:[(0,4.5,[2,2,2,2])],10:[(0,3.75,[0,2,2,2])],11:[(0,4.75,[0,2,2,2])],12:[(0,4.5,[1,2,2,2])]}
notes={1:'Y and T reached with straight arms; arms lower without W.',2:'Clear T and bent-elbow W; diagonal straight-arm Y absent.',3:'Y elbows remain bent; T visible; W uncertain.',4:'Upright low arm movements: Y absent, T partial, W absent.',5:'Y elbows bent, T visible; W uncertain.',6:'Y elbows bent, T visible; W and trunk uncertain in projection.',7:'Only complete second attempt included; Y/T visible, no W.',8:'Complete first attempt Y/T/W visible; truncated second excluded.',10:'Upright T/W visible, no straight diagonal Y.',11:'Provider normal hint not copied: T/W visible but Y absent.',12:'Partial bent-elbow Y; T/W visible.'}
intake=json.loads((B/'output/label-recheck-20261024/new-intake.json').read_text());labels=[]
for r in intake:
 n=int(r['review_id'][1:])
 for a,b,g in spec.get(n,[]):
  assert b<=r['duration']+.01
  labels.append(dict(id=f"recheck_{r['review_id']}_{round(a*1000):06}_{round(b*1000):06}",review_id=r['review_id'],person=r['person'],exercise='ytw',split='train',source=r['path'],sources=[r['source_sha256']],start=a,end=b,fps=r['fps'],grades=g,variant='upright' if n in [4,10] else 'hinged',whole_attempt_observed=True,evidence=notes[n],annotation_status='sampled_4fps_visual_draft',expert_verified=False,provider_label_used_as_target=False))
p=O/'new-labels.frozen.json'
assert not p.exists() or json.loads(p.read_text())==labels
p.write_text(json.dumps(labels,indent=2))
(O/'intake-dispositions.json').write_text(json.dumps(dict(approved_segments=len(labels),approved_sources=len(spec),excluded=dict(N009='Too dark for confident grades',N013_N018='Side shoulder screening did not establish lower-return failure; far arm occluded'),capture_group_alias={'qevd_1948':'capture_shared_rack_1948_370','qevd_370':'capture_shared_rack_1948_370'},review_method='Overview and all twelve dense 4fps YTW sheets inspected; same reviewer, non-expert'),indent=2))
if '--extract' in sys.argv:
 sys.path.insert(0,str(B/'five-exercise-comparison'));import extract
 receipts=[]
 for r in intake:
  if int(r['review_id'][1:]) not in spec:continue
  assert hashlib.file_digest(Path(r['path']).open('rb'),'sha256').hexdigest()==r['source_sha256']
  row=dict(id=r['review_id'],video_sha256=r['source_sha256'],required_views=['front'],segment=dict(start_frame=0,end_frame=r['frame_count']-2,fps=r['fps']))
  rec=extract.extract_view(row,'primary',r['path'],O/'observations');receipts.append(dict(review_id=r['review_id'],source=r['path'],extraction=rec));(O/'extraction.json').write_text(json.dumps(receipts,indent=2));print(r['review_id'],rec['frames'],flush=True)
 print('EXTRACTION_COMPLETE',len(receipts),flush=True)
