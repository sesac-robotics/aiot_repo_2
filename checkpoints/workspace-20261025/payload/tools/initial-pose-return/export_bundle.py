"""Package frozen weights with explicit Train-CV state gates; no Test is read."""
import json,sys,joblib,copy,datetime,os
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=Path(os.environ.get('INITIAL_POSE_RUN',str(B/'output/initial-pose-return-v4-20261024')));sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
bundle=joblib.load(O/'initial-pose-state-candidates.joblib');training=json.loads((O/'state-training.json').read_text());reports={tuple(h['key']):h for h in training['heads'] if h['status']=='trained_experimental'}
for key,h in bundle['heads'].items():
 m=reports[key]['group_cv'];passed=m['balanced_accuracy']>=.8
 if key[2]=='target':passed &= m['nonready_called_ready']/m['nonready_n']<=.15 and m['ready_missed']/m['ready_n']<=.15
 temporal=reports[key].get('temporal_development')
 if temporal is not None:passed &= temporal['passed']
 h.update(state_gate_passed=bool(passed),state_development_metrics=m,not_independent_final_validation=True)
bundle.update(temporal_input='learned_state_class_with_three_observation_dwell_not_uncalibrated_probability_cutoff',trained_contexts={'pushups':'side_arm_return','shoulder':'front_seated_rack','ytw':'upright_Y_T_W','plank':'side_support_hold','squats':'front_standing'},feature_value_parity_verified=True,not_return_severity_grade_model=True,production_replacement=False)
p=O/'initial-pose-return-bundle.joblib';assert not p.exists();joblib.dump(bundle,p);(O/'service-model-freeze.json').write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),model_sha256=sha(p),parent_weights_sha256=sha(O/'initial-pose-state-candidates.joblib'),implementation_sha256=sha(B/'team-posture-scoring/Server/AIService/app/scoring/initial_pose_return.py'),Test_used=False,production_replacement=False),indent=2));print('EXPORTED',sum(h['state_gate_passed'] for h in bundle['heads'].values()),'of',len(bundle['heads']))
