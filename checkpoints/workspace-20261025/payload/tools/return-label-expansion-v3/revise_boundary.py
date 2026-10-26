"""Preserve the original visual draft and explicitly hold its borderline case."""
import json,sys,datetime
from pathlib import Path
B=Path(__file__).resolve().parents[2];D=B/'output/return-label-expansion-v3-20261024'
sys.path.insert(0,str(B/'tools/error-enrichment'));from common import sha
d=json.loads((D/'labels.frozen.json').read_text());held=[];accepted=[]
for r in d['reviewed']:
 if r['id']=='return_v3_E12_000200_001300':
  held.append(dict(r,train_eligible=False,return_grade=None,grades=[None]*4,reason='Original top-every-frame and tracking review: upper elbows are near straight with a small residual bend. Grade1 versus near-straight grade2 is not confident enough for training. Hold, do not relabel to model prediction.'))
 else:accepted.append(r)
d.update(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_labels_sha256=sha(D/'labels.frozen.json'),reviewed=accepted,held=held,new_partial_failure_intervals=1,new_normal_return_intervals=2,revision='Withdraw one borderline first repetition before fitting; retain the second completed visibly bent-arm return draft. No Test/predictions used.')
p=D/'labels-v2.frozen.json';assert not p.exists();p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
s=(B/'tools/return-label-expansion-v3/prepare_scores.py').read_text().replace('return-label-enriched-v3-20261024','return-label-enriched-v4-20261024').replace("D/'labels.frozen.json'","D/'labels-v2.frozen.json'").replace('new_interval_labels=4,new_partial_failure_intervals=2','new_interval_labels=3,new_partial_failure_intervals=1').replace("'added4'","'added3'")
(B/'tools/return-label-expansion-v3/prepare_scores_v4.py').write_text(s)
print('Accepted Train labels: partial1 normal2; borderline1 held before fit')
