"""Exercise the actual combined API on reviewed Train clips, without fitting."""
import json,sys
from pathlib import Path
from fractions import Fraction
import joblib,numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/initial-pose-return-v7-20261024';sys.path[:0]=[str(B/'team-posture-scoring/Server/AIService'),str(B/'tools/error-enrichment')]
from app.scoring.engine import ScoringEngine
from app.scoring.sequence import PoseSequence
from split_guard import POLICY,assert_training_allowed
from common import sha
policy=json.loads(POLICY.read_text());inv={r['id']:r for r in json.loads((B/'output/r2d2-review-20261024/inventory.json').read_text())};labels=json.loads((B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json').read_text())['labels'];state=joblib.load(O/'initial-pose-return-bundle.joblib');current=json.loads((B/'output/CURRENT_FIVE_EXERCISE_RUN.json').read_text());path=Path(current['run'])/'score-candidates.joblib';heads=joblib.load(path)['heads'];engine=ScoringEngine();out=[]
for sid,protocol in [('V28','arm_return_v1'),('V20','seated_rack_v1'),('V01','standing_return_v1'),('V46','support_hold_v1'),('V37','upright_Y_T_W_v1')]:
 r=next(r for r in labels if r['source_id']==sid and r['complete_cycle']);s=inv[sid];assert policy['assignments'][sid]=='train';assert_training_allowed([dict(id=r['id'],person='r2d2_'+r['person'],split='train',sources=[r['source_sha256']])]);p=B/'output/integrated-five-20261024/team-observations'/sid/'side.npz';meta=json.loads(p.with_suffix('.json').read_text());assert sha(s['copy'])==r['source_sha256']==meta['config']['source_sha256'];z=np.load(p);fps=float(Fraction(s['fps']));times=z['frame_numbers']/fps;ix=(times>=r['start'])&(times<r['end']);rot=abs(float(s.get('rotation',[{}])[0].get('rotation',0)))%180;seq=PoseSequence(z['image_xy'][ix],z['frame_numbers'][ix],fps,s['height'] if rot==90 else s['width'],s['width'] if rot==90 else s['height'])
 result=engine.score_with_initial_return(r['exercise'],seq,state_models=state,return_severity_models=heads,protocol=protocol,view=r['view'],complete_attempt=True,permitted_items=[bool(i['training_mask']) for i in r['items']])
 assert result['production_replacement'] is False
 if r['exercise'] in ['pushups','shoulder','squats']:
  assert result['return_grade_integration']['grade'] is None
  assert result['ai_score'] is None
 out.append(dict(source_id=sid,reviewed_interval_id=r['id'],split='train',result=result))
(O/'service-integration-development.json').write_text(json.dumps(dict(completed=True,Test_used=False,models_not_refit=True,score_model_sha256=sha(path),state_model_sha256=sha(O/'initial-pose-return-bundle.joblib'),checks=out,unpassed_or_non_equivalent_return_heads_correctly_held=True,not_a_quality_accuracy_claim=True),indent=2));print('ACTUAL_SERVICE_INTEGRATION',len(out),'reviewed Train clips; unsupported return grades held')
