"""Directly reviewed visible arm-return labels; no provider tags become scores."""
import sys,json,datetime
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/ai-rule-paired-20261024'
sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import assert_training_allowed
intake=json.loads((O/'new-intake.json').read_text())
# Whole completed observed cycles; exclude later kneeling/rest from the attempt.
windows={1:(0,1.6),2:(0,1.9),3:(.3,2.1),4:(.3,3.6),5:(0,2.8),6:(0,3.8),7:(0,1.85),8:(0,2.85),9:(0,3.5),10:(0,2.8),11:(0,2.6),13:(0,2.2),14:(.75,4),15:(0,2.2),17:(0,3.8),18:(0,2.1)}
review=[]
for k,r in enumerate(intake,1):
 n=dict(r,levels=[None]*4,review_method='direct_dense_4fps_visual_review_not_expert_certified',reviewed_dense_sheet_sha256=sha(O/(r['review_id']+'-dense.jpg')),provider_hints_not_grade_labels=True,source_assignment_changed=False)
 if k in windows:
  n.update(start=windows[k][0],end=windows[k][1],levels=[None,None,2,None],train_eligible=True,view='front',variant='visible_arm_return_only',id='paired-return-'+r['id'],notes='Visible arm extension after an observed lowering phase. Whole-body support, trunk and graded depth withheld. Rest/kneeling after cycle is outside window.')
 else:
  n.update(train_eligible=False,notes='Wrist/forearm border ambiguity' if k in [12,16] else 'Oblique shoulder view / far-arm occlusion and insufficient evidence for the four project targets. Native bad tag is not a project score.')
 review.append(n)
assert_training_allowed([dict(r,sources=[r['source_sha256']]) for r in review if r['train_eligible']])
p=O/'reviewed-intake.frozen.json';assert not p.exists();p.write_text(json.dumps(dict(reviewed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),review_ids_are_sources_not_team_person_ids=True,expert_verified=False,labels=review),ensure_ascii=False,indent=2))
print('REVIEW_FROZEN',len(review),'approved',sum(r['train_eligible'] for r in review),flush=True)
