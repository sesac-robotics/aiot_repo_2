import json,sys,datetime
from pathlib import Path
B=Path(__file__).resolve().parents[2];P=B/'output/return-shared-extension-v1-20261024';O=B/'output/return-extension-monotone-v1-20261024';O.mkdir(exist_ok=True)
sys.path.insert(0,str(B/'tools/error-enrichment'));from common import sha
from split_guard import POLICY
for n in ['score-inputs.frozen.json','test-inputs.frozen.json','original-test-inputs.frozen.json','feature-provenance.json']:
 p=O/n;assert not p.exists();p.write_bytes((P/n).read_bytes())
p=O/'MODEL_SELECTION_PROTOCOL.frozen.json';assert not p.exists();p.write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_run=str(P),policy_sha256=sha(POLICY),Train_sha256=sha(O/'score-inputs.frozen.json'),Test_sha256=sha(O/'test-inputs.frozen.json'),hypothesis='Generic unconstrained classifiers can treat a nearly straight arm as an error due to quantile spread or visibility. Require nonnegative coefficients for angular deficits and forbid visibility as a severity cue.',bank='observed_arm_extension_shared_v1',candidate_C=[.1,1.,10.],selection='Train-only nested group CV and <=5% normal-FP priority, followed by error recall and balanced group MAE; threshold from Train group-held probabilities only',diagnostic_Test_previously_exposed_and_used_for_root_cause_analysis=True,independent_Test_claim=False,Test_not_used_in_fit_or_numeric_parameter_selection=True,new_failure_labels=0,production_replacement=False),indent=2))
print('Monotonic anatomical contract frozen before fitting')
