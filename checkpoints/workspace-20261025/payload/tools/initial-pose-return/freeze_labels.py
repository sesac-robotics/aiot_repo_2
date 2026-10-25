"""Human-reviewed static start/return states and Y/T/W phases, never full grades."""
import json,sys,datetime
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/initial-pose-return-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import POLICY,assert_training_allowed
sources={r['id']:r for r in json.loads((O/'source-intake.frozen.json').read_text())};spec={
'V28':([.75,1.,1.25,1.5,4.,4.25],[2.5,2.75,3.,3.25]),
'V30':([0,.25,.5,2.75,3.],[1.,1.25,1.5,1.75]),
'V35':([0,.25,1.75,2.,2.25],[.75,1.,1.25,2.75]),
'V20':([0,3.,3.25,3.5,3.75,4.],[1.,1.25,1.5,1.75]),
'V24':([0,.25,3.,3.25,3.5,3.75],[1.,1.25,1.5,1.75]),
'V26':([0,.25,3.,3.25,3.5,3.75],[1.,1.25,1.5,1.75]),
'V01':([0,.25,3.],[1.25,1.5,1.75,2.,2.25]),
'V07':([0,.25,.5,.75,3.,3.25,3.5],[1.5,1.75,2.,2.25]),
'V13':([0,1.75,2.,2.25,4.75,5.,5.25],[.75,1.,3.,3.25,3.5]),
'V17':([0,.25,3.75,4.,4.25],[1.5,2.,2.5,2.75]),
'V49':([],[0,.25,.5,1.,2.,3.,4.]),
'V46':([0,.25,.5,1.,2.,3.,4.],[]),
'V47':([],[0,.25,.5,1.,2.,3.,4.]),
'V48':([0,.25,.5,1.,2.,3.,4.],[]),
}
phases={
'V37':{'Y':[0,3.75,4.,4.25],'T':[.5,.75,1.,4.75,5.,5.25],'W':[1.75,2.,2.25,2.5,3.,6.,6.25,6.5]},
'V39':{'Y':[0,.25,.5,4.25,4.5,4.75],'T':[1.,1.25,1.5,6.5,6.75,7.],'W':[2.,2.25,2.5,3.,7.75,8.,8.25]},
'V41':{'Y':[0,.25,.5,4.,4.25,4.5],'T':[1.,1.25,1.5,5.25,5.5,5.75],'W':[2.,2.25,2.5,3.,6.5,6.75,7.]},
'V43':{'Y':[0,.25,.5,.75,1.,5.75,6.,6.25],'T':[2.,2.25,2.5,2.75,7.25,7.5,7.75],'W':[3.75,4.,4.25,4.5,8.75,9.,9.25]}}
policy=json.loads(POLICY.read_text());records=[]
for src in sources.values():
 assert policy['assignments'][src['id']]=='train';sheet=B/'output/r2d2-review-20261024'/(src['id']+'-dense-01.jpg');assert sheet.exists()
 if src['id'] in phases:pairs=[(t,int(k=='Y'),k) for k,ts in phases[src['id']].items() for t in ts]
 else:pairs=[(t,y,None) for y,ts in [(1,spec[src['id']][0]),(0,spec[src['id']][1])] for t in ts]
 for t,y,phase in pairs:
  assert t<src['duration']
  records.append(dict(id=f"{src['id']}_{round(t*1000):06}",source_id=src['id'],exercise=src['exercise'],person='r2d2_'+src['person'],group='r2d2_'+src['person'],split='train',sources=[src['source_sha256']],target=y,phase=phase,review_time=t,source_sha256=src['source_sha256'],review_sheet=str(sheet),review_sheet_sha256=sha(sheet),review_status='direct_sampled_visual_draft_not_expert_verified',not_ready_frame_is_not_repetition_failure=True,overall_quality_grade=None))
assert_training_allowed(records);p=O/'state-labels.frozen.json';assert not p.exists();p.write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),labels=sorted(records,key=lambda r:(r['source_id'],r['review_time'])),target_semantics='Exercise-specific initial/return item state; YTW startY and separate Y/T/W phases; plank support hold state',no_grade_zero_inferred=True,team_rep_labels_and_partitions_unchanged=True,policy_sha256=sha(POLICY)),ensure_ascii=False,indent=2));print('FRAME_LABELS',len(records),flush=True)
