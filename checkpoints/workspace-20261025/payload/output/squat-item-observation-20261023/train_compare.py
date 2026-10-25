"""Frozen labels, identical held people and sources; fit boundaries on train only."""
import json,hashlib,collections
from pathlib import Path
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
O=Path(__file__).resolve().parent;rows=json.loads((O/'feature-rows.json').read_text());heads={};reports={}
# One global person partition shared by all heads and before/after candidates.
groups=np.array([r['person'] for r in rows]);folds=[]
for number,(tr,te) in enumerate(GroupKFold(5).split(rows,groups=groups)):
 train=[rows[i] for i in tr];held=[rows[i] for i in te]
 assert not {r['person'] for r in train}&{r['person'] for r in held}
 assert not {s for r in train for s in r['source']}&{s for r in held for s in r['source']}
 folds.append(dict(number=number,train_people=sorted({r['person'] for r in train}),held_people=sorted({r['person'] for r in held}),train_ids=[r['id'] for r in train],held_ids=[r['id'] for r in held]))
def rule(v,b,higher):return 2 if (v>=b[0] if higher else v<=b[0]) else 1 if (v>=b[1] if higher else v<=b[1]) else 0
def fit(rs,key,which):
 y=np.array([r['levels'][key] for r in rs]);support=collections.Counter(y);weights=np.array([1/support[v] for v in y]);x=[r[which][key]['x'] for r in rs]
 model=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=2,max_features=1.,random_state=23,n_jobs=2).fit(x,y,sample_weight=weights)
 values=sorted(set(r[which][key]['scalar'] for r in rs));cuts=[values[0]-.01]+[(a+b)/2 for a,b in zip(values,values[1:])]+[values[-1]+.01];higher=rs[0][which][key]['higher'];options=[]
 for g in cuts:
  for p in cuts:
   if (higher and g<=p) or (not higher and g>=p):continue
   loss=np.average(abs(np.array([rule(r[which][key]['scalar'],[g,p],higher) for r in rs])-y),weights=weights);options.append((float(loss),g,p))
 _,g,p=min(options)
 return dict(model=model,boundaries=[g,p],higher=higher,train_ids=[r['id'] for r in rs],training_people=sorted({r['person'] for r in rs}),train_support={str(k):v for k,v in support.items()})
def metrics(ps,field):
 selected=[r for r in ps if r.get(field) is not None];ys=np.array([r['target'] for r in selected]);vs=np.array([r[field] for r in selected]);err=ys<2;normal=ys==2
 return dict(n=len(ys),normal_n=int(normal.sum()),error_n=int(err.sum()),MAE=float(np.mean(abs(vs-ys))) if len(ys) else None,normal_flagged=int(((vs<1.5)&normal).sum()),missed_errors=int(((vs>=1.5)&err).sum()),error_MAE=float(np.mean(abs(vs[err]-ys[err]))) if err.any() else None,balanced_MAE=float(np.mean([np.mean(abs(vs[ys==g]-g)) for g in sorted(set(ys))])) if len(ys) else None)
for key in ['knee_inward','trunk_balance']:
 preds=[]
 for f in folds:
  train=[r for r in rows if r['id'] in f['train_ids']];held=[r for r in rows if r['id'] in f['held_ids']]
  before=[r for r in train if r['origin']=='legacy' and r['levels'].get(key) is not None and key in r['before']]
  after=[r for r in train if r['levels'].get(key) is not None and key in r['after']]
  bm=fit(before,key,'before');am=fit(after,key,'after')
  for r in held:
   if r['levels'].get(key) is None or key not in r['after']:continue
   af=r['after'][key];out=dict(id=r['id'],person=r['person'],origin=r['origin'],fold=f['number'],target=r['levels'][key],before_ai=None,before_rule=None,after_ai=float(np.clip(am['model'].predict([af['x']])[0],0,2)),after_rule=rule(af['scalar'],am['boundaries'],am['higher']))
   if key in r['before']:
    bf=r['before'][key];out.update(before_ai=float(np.clip(bm['model'].predict([bf['x']])[0],0,2)),before_rule=rule(bf['scalar'],bm['boundaries'],bm['higher']))
   preds.append(out)
 common=[r for r in preds if r['before_ai'] is not None];legacy=[r for r in common if r['origin']=='legacy'];new=[r for r in common if r['origin']=='recovered'];result={}
 for name,ps in [('common',common),('legacy',legacy),('recovered',new)]:
  result[name]={method:{'before':metrics(ps,'before_'+method),'after':metrics(ps,'after_'+method)} for method in ['ai','rule']}
 result.update(predictions=preds,coverage_gained=[r['id'] for r in preds if r['before_ai'] is None],new_training_items=sum(r['origin']=='recovered' and r['levels'].get(key) is not None and key in r['after'] for r in rows))
 # Predeclared conservative deployment gate: no more missed errors, no more
 # normal flags, strictly lower balanced MAE on common+legacy cases.
 def improved(method):
  good=True;strict=False
  for scope in ['common','legacy']:
   b=result[scope][method]['before'];a=result[scope][method]['after'];good &= a['missed_errors']<=b['missed_errors'] and a['normal_flagged']<=b['normal_flagged'] and a['balanced_MAE']<=b['balanced_MAE'];strict |= a['balanced_MAE']<b['balanced_MAE']
  return bool(good and strict)
 result['development_gate']={m:improved(m) for m in ['ai','rule']}
 # Candidate only, never replace the deployed models based on this reused dev set.
 eligible=[r for r in rows if r['levels'].get(key) is not None and key in r['after']];heads[key]=fit(eligible,key,'after');reports[key]=result
print(json.dumps({key:{k:v for k,v in value.items() if k not in ['predictions']} for key,value in reports.items()},ensure_ascii=False),flush=True)
labelsha=hashlib.sha256((O/'labels.frozen.json').read_bytes()).hexdigest();joblib.dump(dict(version='item-observation-recovered-candidate-v1',heads=heads,label_sha256=labelsha,runtime_approved=False),O/'candidate.joblib')
return_support=collections.Counter(r['levels'].get('standing_return') for r in rows if r['levels'].get('standing_return') is not None and 'standing_return' in r['after'])
report=dict(folds=folds,items=reports,return_support=dict(return_support),return_AI_trained=False,return_reason='정상만 증가했으며 복귀 오류 라벨이 없어 AI를 훈련하지 않음',candidate_sha256=hashlib.sha256((O/'candidate.joblib').read_bytes()).hexdigest(),label_sha256=labelsha,independent_final_test=False,runtime_models_replaced=False,limitation='기존 개발 자료를 재사용한 사람별 분리 검증. 직접 시각 표본 라벨 초안이며 전문가 판정·독립 최종 시험이 아님.')
(O/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
