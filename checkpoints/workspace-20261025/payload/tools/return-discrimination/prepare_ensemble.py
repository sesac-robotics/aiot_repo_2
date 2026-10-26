import json,datetime,sys
from pathlib import Path
B=Path(__file__).resolve().parents[2];P=B/'output/return-discrimination-v2-20261024';O=B/'output/return-discrimination-ensemble-20261024';O.mkdir(exist_ok=True);sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
for n in ['score-inputs.frozen.json','test-inputs.frozen.json','original-test-inputs.frozen.json','feature-provenance.json']:
 assert not (O/n).exists();(O/n).write_bytes((P/n).read_bytes())
p=json.loads((P/'MODEL_SELECTION_PROTOCOL.frozen.json').read_text());p.update(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_experiment=str(P),model_type='leave-one-Train-participant-group-out positive-deficit ensemble',calibration='OOF cutoff and inference members have the same training group count; no full-data refit after OOF calibration',candidate_C=[.1,1.,10.],normal_false_alarm_cap=.05,final_Test_opened=False,Test_used_for_selection=False,model_code_sha256=sha(B/'tools/error-enrichment/models.py'));(O/'MODEL_SELECTION_PROTOCOL.frozen.json').write_text(json.dumps(p,indent=2))
