import json,datetime,sys
from pathlib import Path
B=Path(__file__).resolve().parents[2];P=B/'output/return-discrimination-v1-20261024';O=B/'output/return-discrimination-v2-20261024';O.mkdir(exist_ok=True);sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
for n in ['score-inputs.frozen.json','test-inputs.frozen.json','original-test-inputs.frozen.json','feature-provenance.json']:
 assert not (O/n).exists();(O/n).write_bytes((P/n).read_bytes())
protocol=json.loads((P/'MODEL_SELECTION_PROTOCOL.frozen.json').read_text());protocol.update(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_experiment=str(P),normal_false_alarm_cap=.05,selection='Train nested group validation only: prioritize <=5 percent normal FP; then recall>=.65; then mild error recall; then total recall; then normal FP; then balanced MAE',reason='User explicitly requires reducing normal false alarms; no development IDs/labels/cutoff values used for selection',independent_validation_claim=False)
(O/'MODEL_SELECTION_PROTOCOL.frozen.json').write_text(json.dumps(protocol,indent=2));s=(B/'tools/return-discrimination/train.py').read_text().replace('output/return-discrimination-v1-20261024','output/return-discrimination-v2-20261024').replace("m['normal_false_alarm']>.15","m['normal_false_alarm']>.05");s=s.replace("not(met['error_recall']>=.65 and met['normal_false_alarm']<=.15 and len(mp)==len(mild))","met['normal_false_alarm']>.05,met['error_recall']<.65 or len(mp)!=len(mild)");(B/'tools/return-discrimination/train_strict.py').write_text(s)
