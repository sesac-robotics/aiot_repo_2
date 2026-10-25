"""Compare label expansion and optional stance context on identical held people."""
import json,collections,hashlib,sys
from pathlib import Path
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesRegressor
O=Path(__file__).resolve().parent;B=O.parents[1];rows=json.loads((O/'feature-rows.json').read_text());folds=json.loads((B/'output/squat-item-observation-20261023/comparison.json').read_text())['folds'];reports={};bundles={}
def rule(v,b,h):return 2 if (v>=b[0] if h else v<=b[0]) else 1 if (v>=b[1] if h else v<=b[1]) else 0
def fit(rs,key,feature='core'):
 y=np.array([r['levels'][key] for r in rs]);n=collections.Counter(y);w=np.array([1/n[v] for v in y]);model=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=2,max_features=1.,random_state=23,n_jobs=2).fit([r[feature][key]['x'] for r in rs],y,sample_weight=w)
 vals=sorted(set(r[feature][key]['scalar'] for r in rs));cuts=[vals[0]-.01]+[(a+b)/2 for a,b in zip(vals,vals[1:])]+[vals[-1]+.01];h=rs[0][feature][key]['higher'];opts=[]
 for g in cuts:
  for p in cuts:
   if (h and g<=p) or (not h and g>=p):continue
   loss=float(np.average(abs(np.array([rule(r[feature][key]['scalar'],[g,p],h) for r in rs])-y),weights=w));opts.append((loss,g,p))
 _,g,p=min(opts);return dict(model=model,boundaries=[g,p],higher=h,train_support={str(k):v for k,v in n.items()},train_ids=[r['id'] for r in rs],training_people=sorted({r['person'] for r in rs}))
def metrics(ps,field):
 y=np.array([p['target'] for p in ps]);v=np.array([p[field] for p in ps]);err=y<2;normal=y==2
 return dict(n=len(y),normal_n=int(normal.sum()),error_n=int(err.sum()),normal_flagged=int(((v<1.5)&normal).sum()),missed_errors=int(((v>=1.5)&err).sum()),MAE=float(np.mean(abs(v-y))),error_MAE=float(np.mean(abs(v[err]-y[err]))) if err.any() else None,balanced_MAE=float(np.mean([np.mean(abs(v[y==g]-g)) for g in sorted(set(y))])))
def eligible(rs,key):return [r for r in rs if r['levels'].get(key) is not None and key in r['core']]
def fit_heads(rs,key):
 core=fit(rs,key);stance=[r for r in rs if key=='knee_inward' and r['stance']]
 return dict(core=core,stance=fit(stance,key,'optional') if stance else None)
def predict(head,r,key,context=False):
 feature='optional' if context and key=='knee_inward' and r['stance'] and head.get('stance') else 'core';h=head['stance'] if feature=='optional' else head['core'];f=r[feature][key];return float(np.clip(h['model'].predict([f['x']])[0],0,2)),rule(f['scalar'],h['boundaries'],h['higher'])
for key in ['knee_inward','trunk_balance']:
 ps=[]
 for fold in folds:
  train=[r for r in rows if r['person'] in fold['train_people']];held=eligible([r for r in rows if r['person'] in fold['held_people']],key)
  assert not {r['person'] for r in train}&{r['person'] for r in held};assert not {s for r in train for s in r['source']}&{s for r in held for s in r['source']}
  base=fit_heads(eligible([r for r in train if r['old_corpus']],key),key);after=fit_heads(eligible(train,key),key)
  for r in held:
   ba,br=predict(base,r,key);aa,ar=predict(after,r,key);ca,cr=predict(after,r,key,True)
   ps.append(dict(id=r['id'],person=r['person'],target=r['levels'][key],old_corpus=r['old_corpus'],orientation=r['orientation'],fold=fold['number'],before_ai=ba,before_rule=br,added_ai=aa,added_rule=ar,context_ai=ca,context_rule=cr,stance_observed=r['stance'] is not None))
 stats={}
 for name,subset in [('all',ps),('legacy',[p for p in ps if p['old_corpus']]),('near_front_new',[p for p in ps if not p['old_corpus']])]:
  stats[name]={mode:{m:metrics(subset,mode+'_'+m) for m in ['ai','rule']} for mode in ['before','added','context']}
 # An improvement must not trade missed errors for lower normal false positives.
 gate={}
 for mode in ['added','context']:
  gate[mode]={}
  for method in ['ai','rule']:
   ok=True;strict=False
   for scope in ['all','legacy']:
    a=stats[scope][mode][method];b=stats[scope]['before'][method];ok &= a['missed_errors']<=b['missed_errors'] and a['normal_flagged']<=b['normal_flagged'] and a['balanced_MAE']<=b['balanced_MAE'];strict |= a['balanced_MAE']<b['balanced_MAE']
   gate[mode][method]=bool(ok and strict)
 reports[key]=dict(metrics=stats,predictions=ps,development_gate=gate)
 # Candidate fits exclude Mima, which remains the existing validation person.
 train=eligible([r for r in rows if r['person']!='rehab26_Mima'],key);bundles[key]=fit_heads(train,key)
 print(key,json.dumps(dict(stats=stats,gate=gate),ensure_ascii=False),flush=True)
sys.path.insert(0,str(B/'output/live-webcam-20261023'));from checklist_comparison import models
active,_=models();fixed={}
for key in ['knee_inward','trunk_balance']:
 head,path,sha=active[key];ps=[]
 for r in eligible([r for r in rows if r['person']=='rehab26_Mima'],key):
  f=r['core'][key];ba=float(np.clip(head['model'].predict([f['x']])[0],0,2));br=rule(f['scalar'],head['boundaries'],head['higher']);aa,ar=predict(bundles[key],r,key);ca,cr=predict(bundles[key],r,key,True)
  ps.append(dict(id=r['id'],person=r['person'],target=r['levels'][key],before_ai=ba,before_rule=br,added_ai=aa,added_rule=ar,context_ai=ca,context_rule=cr))
 fixed[key]=dict(metrics={mode:{m:metrics(ps,mode+'_'+m) for m in ['ai','rule']} for mode in ['before','added','context']},predictions=ps)
 assert 'Mima' not in head['training_people'] and 'rehab26_Mima' not in bundles[key]['core']['training_people']
labelsha=hashlib.sha256((O/'labels.frozen.json').read_bytes()).hexdigest();joblib.dump(dict(version='error-strengthening-v1',heads=bundles,validation_person='rehab26_Mima',labels_sha256=labelsha,runtime_approved=False),O/'candidate.joblib')
report=dict(items=reports,fixed_validation=fixed,folds=folds,labels_sha256=labelsha,candidate_sha256=hashlib.sha256((O/'candidate.joblib').read_bytes()).hexdigest(),return_AI_trained=False,independent_final_test=False,original_test_person_opened=False,candidate_runtime_deployed=False,notes=['이전 개발 검증 재사용이며 독립 최종 시험 아님','근정면10·20도 라벨을 추가했으며 정면 기존자료 성능을 별도 확인','발목 부가 모델은 관측될 때만 사용. 없으면 6특징 기본 모델로 복귀','무릎 간격 감소와 발목 대비 안쪽 정렬은 동일 현상이 아님. 부가 모델을 기존 점수에 자동 대체하지 않음'])
(O/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('fixed',json.dumps({k:v['metrics'] for k,v in fixed.items()},ensure_ascii=False),flush=True)
