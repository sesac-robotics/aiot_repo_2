import json,sys,datetime
from pathlib import Path
B=Path(__file__).resolve().parents[2];P=B/'output/return-discrimination-v2-20261024';O=B/'output/return-discrimination-mean-20261024';O.mkdir(exist_ok=True);sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
for n in ['score-inputs.frozen.json','test-inputs.frozen.json']:
 rows=json.loads((P/n).read_text())
 for r in rows:
  for i in r['items']:
   if i['bank']=='standing_support_extension_v1':
    i['bank']='standing_support_extension_mean_v1';i['feature_version']='standing-support-extension-mean-v1'
    if i['feature'] is not None:i['feature']=i['feature'][::2]
 assert not (O/n).exists();(O/n).write_text(json.dumps(rows,indent=2))
(O/'original-test-inputs.frozen.json').write_bytes((P/'original-test-inputs.frozen.json').read_bytes());protocol=json.loads((P/'MODEL_SELECTION_PROTOCOL.frozen.json').read_text());protocol.update(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),previous_experiment=str(P),new_bank='standing_support_extension_mean_v1',feature_names=protocol['feature_names'][::2],feature_ablation='side average only; remove maximum-side features that amplify one-side projection noise',Test_used_for_selection=False,code_sha256=sha(Path(__file__)));(O/'MODEL_SELECTION_PROTOCOL.frozen.json').write_text(json.dumps(protocol,indent=2));s=(B/'tools/return-discrimination/train_strict.py').read_text().replace('output/return-discrimination-v2-20261024','output/return-discrimination-mean-20261024').replace('standing_support_extension_v1','standing_support_extension_mean_v1').replace('feature_dimensions=10','feature_dimensions=5');(B/'tools/return-discrimination/train_mean.py').write_text(s)
