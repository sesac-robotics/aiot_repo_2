import json,datetime
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/ai-rule-paired-20261024'
r=json.loads((O/'reviewed-intake.frozen.json').read_text());windows={1:(0,1.55),2:(0,1.8),3:(.3,2.05),4:(.3,3.3),5:(0,2.65),6:(0,3.95),7:(0,1.65),8:(0,2.9),9:(0,3.65),10:(0,3.0),11:(0,2.4),13:(0,2.3),14:(1.1,4.85),15:(0,2.4),17:(0,4.3),18:(0,1.95)}
for k,n in enumerate(r['labels'],1):
 if k in windows:n.update(start=windows[k][0],end=windows[k][1],window_revision='end_phase_must_remain_extended_v2')
r.update(reviewed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),supersedes='reviewed-intake.frozen.json',revision_reason='Existing return descriptor reads terminal frames. Remove next partial lowering from normal-return windows.')
p=O/'reviewed-intake-v2.frozen.json';assert not p.exists();p.write_text(json.dumps(r,ensure_ascii=False,indent=2))
(B/'output/paired-enriched-five-20261024/REJECTED_PREPARATION.json').write_text(json.dumps(dict(status='superseded_not_valid_for_evaluation_or_promotion',reason='Some normal-return windows included next descent in last frames.',production_replacement=False,only_trial_return_head_fitted=True),indent=2))
