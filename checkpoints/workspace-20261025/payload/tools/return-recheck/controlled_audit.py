import json,sys
from pathlib import Path
import joblib
B=Path(__file__).resolve().parents[2];O=B/'output/return-initial-context-20261024';P=B/'output/return-discrimination-ensemble-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'))
from train_scores import metrics
read=lambda p:json.loads(p.read_text());h=joblib.load(P/'score-candidates.joblib')['heads'][('squats','standing_height_return','standing_support_extension_v1')];old=read(P/'new-head-training.json')['validation_predictions'];newrows={r['id']:r for r in read(B/'output/return-context-20261024/score-inputs.frozen.json')};pred=[]
for p in old:
 r=newrows[p['id']];i=next(i for i in r['items'] if i['bank']=='standing_support_context_v1');assert i['eligible'];pred.append(dict(p,prediction=float(h['model'].predict([i['feature']])[0])))
result=dict(control='same old fitted model, same old reference, same same17 IDs/targets, only bounded-end-context features changed',new_fit=False,old_features=metrics(old),corrected_context_features=metrics(pred),predictions=pred,original_labels_unchanged=True);(O/'timing-only-control.json').write_text(json.dumps(result,indent=2));print(result['old_features'],result['corrected_context_features'])
