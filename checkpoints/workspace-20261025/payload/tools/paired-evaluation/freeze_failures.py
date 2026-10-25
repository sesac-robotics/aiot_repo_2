import json,sys,datetime
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/paired-failure-mining-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'))
from split_guard import assert_training_allowed
from common import sha
rs=json.loads((O/'intake-v2.frozen.json').read_text())['records']+json.loads((O/'upper-intake.frozen.json').read_text())+json.loads((O/'lateral-intake.frozen.json').read_text());spec={
'F001':(0,1.4,[None,None,1,None]),'F002':(.9,3.,[None,None,2,None]),'F003':(.8,2.7,[None,None,1,None]),'F004':(0,2.3,[None,None,2,None]),'F006':(.8,3.35,[None,None,1,None]),'F007':(.85,3.8,[None,None,2,None]),'F008':(0,2.15,[None,None,1,None]),
'F013':(0,2.3,[1,2,None,None]),'F014':(0,2.45,[1,2,None,None]),'F015':(0,2.7,[1,2,None,None]),
'U001':(0,4.,[1,2,None,None]),'U002':(0,4.1,[1,2,None,None]),'U003':(0,4.2,[1,2,None,None]),'U004':(0,3.7,[1,2,None,None]),'U005':(6.5,10.2,[2,2,1,None])}
for r in rs:
 rid=r['review_id'];r.update(train_eligible=rid in spec,manual_review='dense_4fps_direct_visual_draft_not_expert_certified',provider_hint_not_used_as_grade=True)
 if rid in spec:
  a,b,g=spec[rid];r.update(start=a,end=b,grades=g,split='train',id='failure-expansion-'+rid+'-'+r['id'],family='MultiView' if rid=='U005' else 'QEVD',view='front',variant='hinged' if r['exercise']=='ytw' else 'front_seated' if r['exercise']=='shoulder' else 'arm_return_only' if r['exercise']=='pushups' else 'bodyweight_front',whole_attempt_observed=True)
  if r['exercise']=='pushups':r['review_note']='Repeated top phase visibly remains elbow-flexed (grade1) or extends (grade2). Body contact/variant correctness, depth, trunk ungraded; partial-return failures are not end-of-video truncation or kneeling after completion.'
  elif r['exercise']=='squats':r['review_note']='Visible shallow squat, followed by standing at original height. First0.0-0.2s verified standing reference. Knee/trunk targets held.';r['verified_standing_reference_seconds']=[0,.1,.2]
  elif r['exercise']=='ytw':r['review_note']='Within an observed Y-to-T-to-lower sequence, Y arm raise stays low/elbows bent, T reaches side plane. W severity and sagittal hinge alignment held.'
  else:r['review_note']='Both arms reach overhead and lower to shoulder level; visible left-right timing offset during descent. Frontal trunk target held.'
  r['dense_sheet_sha256']=sha(O/(rid+'-dense.jpg'))
 else:r.update(grades=[None]*4,review_note='Cannot verify support/ground contact from front upper-body framing' if r['exercise']=='plank' else 'Ambiguous top extension/asymmetry boundary; do not force grade')
assert_training_allowed([dict(r,sources=[r['source_sha256']]) for r in rs if r['train_eligible']]);p=O/'labels.frozen.json';assert not p.exists();p.write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),expert_verified=False,reviewed=rs),ensure_ascii=False,indent=2));print('LABELS',len(rs),'approved',len(spec),flush=True)
