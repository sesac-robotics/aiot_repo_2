from pathlib import Path
import json,sys,collections,numpy as np
O=Path(__file__).resolve().parent
# Read definitions without executing the previous audit's training loop.
s=(O.parent/'squat-errors-20261023/train_rule_focus.py').read_text();ns={'__file__':str(O.parent/'squat-errors-20261023/train_rule_focus.py')};exec(s[:s.index('for h in prior')],ns)
r=json.loads((O/'axis-comparison.json').read_text());rows=r['cases'];ps=[];folds=[]
for f in r['folds']:
 train=[a for a in rows if a['person'] in f['train_people']];held=[a for a in rows if a['id'] in f['held_ids']]
 for a in train:
  a['features']['axis']={'higher':False,'scalar':a['phase_features'][15] if a['phase_features'][-1] else 0};a['levels']['axis']=a['levels']['knee_inward']
 bounds,higher=ns['fit'](train,'axis',1)
 for a in held:
  pred=ns['predict'](a['phase_features'][15],bounds,higher) if a['phase_features'][-1] else ns['predict'](a['features']['knee_inward']['scalar'],*ns['fit'](train,'knee_inward'))
  ps.append(dict(id=a['id'],person=a['person'],fold=f['fold'],target=a['levels']['knee_inward'],prediction=pred))
 folds.append(dict(fold=f['fold'],bounds=bounds,train_people=f['train_people'],held_people=f['held_people']))
result=dict(after=ns['metrics'](ps),predictions=ps,folds=folds,independent_final_test=False,runtime_approved=False)
(O/'axis-rule-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(result['after']);print([p for p in ps if p['target']<2])
