"""Compare frozen previous candidate vs expanded candidate on identical held-out-person clips."""
from pathlib import Path
import json,hashlib,sys
import joblib,numpy as np
P=Path(__file__).parent;old=P.parent/'squat-relabel-20261023'
sys.path.insert(0,str(P.parent/'live-webcam-20261023'))
from runtime_snapshot import digest,read_source
from criteria_features import observed_features

def rule(v,h):
 good,partial=h['boundaries'];return 2 if (v>=good if h['higher'] else v<=good) else 1 if (v>=partial if h['higher'] else v<=partial) else 0
before=joblib.load(old/'candidate.joblib')['heads'];after=joblib.load(P/'candidate.joblib')['heads'];rows=json.loads((P/'feature-rows.json').read_text());out={}
for key in before.keys() & after.keys():
 cases=[]
 for r in rows:
  if r['split']!='validation' or r['levels'][key] is None or key not in r['features']:continue
  f=r['features'][key];case=dict(id=r['id'],target=r['levels'][key])
  for name,heads in [('before',before),('after',after)]:
   h=heads[key];case[name]=dict(ai=float(np.clip(h['model'].predict([f['x']])[0],0,2)),rule=rule(f['scalar'],h))
  cases.append(case)
 summaries={}
 for group,subset in [('original_validation_5',[r for r in cases if int(r['id'].split('_')[1])<=5]),('expanded_validation_10',cases)]:
  summaries[group]=dict(cases=len(subset),**{name:{mode:float(np.mean([abs(c[name][mode]-c['target']) for c in subset])) for mode in ['ai','rule']} for name in ['before','after']})
 out[key]=dict(comparison=summaries,cases=cases)
result=dict(items=out,baseline='previous isolated 20-clip candidate, not original team AI',same_people_split=True,test_opened=False,expert_validated=False,team_source_digest=digest(read_source()),previous_candidate_sha256=hashlib.sha256((old/'candidate.joblib').read_bytes()).hexdigest(),new_candidate_sha256=hashlib.sha256((P/'candidate.joblib').read_bytes()).hexdigest())
(P/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:v['comparison'] for k,v in out.items()},indent=2))
