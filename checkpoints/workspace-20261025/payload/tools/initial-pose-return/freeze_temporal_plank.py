"""Freeze visually reviewed constant plank windows before candidate comparison."""
import json,sys,shutil
from pathlib import Path
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');D=B/'output/initial-pose-return-v6-20261024';O=B/'output/initial-pose-return-v7-20261024'
sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import assert_training_allowed
O.mkdir();names=['state-labels-v3.frozen.json','source-intake.frozen.json','external-intake-audit.json']
for n in names:shutil.copyfile(D/n,O/n)
labels=json.loads((O/names[0]).read_text())['labels'];windows=[]
for s in json.loads((O/names[1]).read_text()):
 if s['exercise']!='plank' or s['id']=='V47':continue
 rs=[r for r in labels if r['source_id']==s['id']];targets={r['target'] for r in rs};assert len(targets)==1
 cp=Path(s['cache']) if s.get('cache') else B/'output/integrated-five-20261024/team-observations'/s['id']/'side.npz'
 windows.append(dict(id=s['id']+'_constant_hold',source_id=s['id'],person=rs[0]['person'],split='train',sources=rs[0]['sources'],source_sha256=s['source_sha256'],cache=str(cp),cache_sha256=sha(cp),fps=float(s['fps'].split('/')[0])/float(s['fps'].split('/')[1]) if isinstance(s['fps'],str) and '/' in s['fps'] else float(s['fps']),start=s['review_start'],end=s['review_end'],target=next(iter(targets)),review_sheet=rs[0]['review_sheet'],review_sheet_sha256=rs[0]['review_sheet_sha256'],review_status='rechecked_chronological_constant_pose_window_nonexpert',failure_grade=None))
assert_training_allowed(windows)
(O/'plank-temporal-development.frozen.json').write_text(json.dumps(dict(windows=windows,scope='Train_person_excluded_development_model_selection_not_independent_Test',V47_contact_uncertain_held=True),indent=2));print('TEMPORAL_WINDOWS',len(windows))
