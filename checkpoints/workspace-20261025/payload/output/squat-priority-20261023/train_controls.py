"""Compare frozen baselines and augmentation with identical held people/cases."""
from pathlib import Path
import sys,json,hashlib,collections,importlib.util
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent;S=R/'output/squat-relabel-expanded-20261023';L=R/'output/live-webcam-20261023';sys.path.insert(0,str(L))
from runtime_snapshot import read_source,digest
from accepted_items import load
source_sha=digest(read_source());_,extract=load();raw=(O/'labels.frozen.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==(O/'labels.sha256').read_text()
old=[]
for r in json.loads((S/'labels.frozen.json').read_text())['records']:
 z=np.load(S/'observations'/r['id']/'front.npz');old.append(dict(id=r['id'],person='rehab26_'+r['person'],split=r['split'],features=extract(z['image_xy']),levels=r['levels'],source=r['sources'],origin='existing_visual_labels'))
new=[]
for r in json.loads(raw)['labels']:
 z=np.load(O/'observations'/r['id']/'front.npz');x=z['image_xy'].copy();outside=((x[:,:,0]<0)|(x[:,:,0]>=z['width'])|(x[:,:,1]<0)|(x[:,:,1]>=z['height']));x[outside]=np.nan
 new.append(dict(id=r['id'],person=r['person'],split='train',features=extract(x),levels=dict(trunk_balance=r['trunk_balance'],knee_inward=r['knee_inward']),source=[r['source_sha256']],origin='priority_normal_controls'))
prior_knee=joblib.load(R/'output/squat-relabel-20261023/candidate.joblib');prior_body=joblib.load(S/'candidate.joblib');heads={};reports=[]
def fit(rows,key):
 y=[r['levels'][key] for r in rows];counts=collections.Counter(y);m=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=2,max_features=1.,random_state=23,n_jobs=2)
 return m.fit([r['features'][key]['x'] for r in rows],y,sample_weight=[1/counts[g] for g in y])
def rule(v,b,higher):
 good,partial=b;return 2 if (v>=good if higher else v<=good) else 1 if (v>=partial if higher else v<=partial) else 0

def fit_rule(rows,key,prior):
 higher=rows[0]['features'][key]['higher'];v=sorted(set(r['features'][key]['scalar'] for r in rows));cuts=[v[0]-.01]+[(a+b)/2 for a,b in zip(v,v[1:])]+[v[-1]+.01]
 if len(cuts)>32:cuts=list(np.quantile(cuts,np.linspace(0,1,32)))
 counts=collections.Counter(r['levels'][key] for r in rows);options=[tuple(prior)]
 for a in cuts:
  for b in cuts:
   if (higher and a>b) or (not higher and a<b):options.append((a,b))
 def loss(b):return np.average([abs(rule(r['features'][key]['scalar'],b,higher)-r['levels'][key]) for r in rows],weights=[1/counts[r['levels'][key]] for r in rows])
 return list(min(options,key=lambda b:(loss(b),tuple(b)!=tuple(prior))))

def metrics(preds,name):
 if not preds:return None
 y=np.array([p['target'] for p in preds]);p=np.array([p[name] for p in preds]);levels=sorted(set(y));by={str(int(g)):float(np.mean(abs(p[y==g]-g))) for g in levels}
 return dict(n=len(preds),MAE=float(np.mean(abs(p-y))),balanced_MAE=float(np.mean(list(by.values()))),grade_MAE=by,normal_n=int(sum(y==2)),error_n=int(sum(y<2)),normal_false_warning=float(np.mean(p[y==2]<1.5)) if any(y==2) else None,error_MAE=float(np.mean(abs(p[y<2]-y[y<2]))) if any(y<2) else None,error_missed=float(np.mean(p[y<2]>=1.5)) if any(y<2) else None)
def no_regression(a,b):
 return a['MAE']<=b['MAE']+1e-8 and a['balanced_MAE']<b['balanced_MAE']-1e-8 and a['error_MAE']<=b['error_MAE']+1e-8 and a['normal_false_warning']<=b['normal_false_warning']+1e-8
for key,prior_bundle in [('knee_inward',prior_knee),('trunk_balance',prior_body)]:
 baseline=prior_bundle['heads'][key];prior_bounds=baseline['boundaries'];higher=baseline['higher'];existing=[r for r in old if r['levels'].get(key) is not None and key in r['features']];added=[r for r in new if r['levels'].get(key) is not None and key in r['features']];cases=existing+added
 # Same people across all recordings are held together. No REHAB24 is mixed,
 # resolving the earlier cross-REHAB family ambiguity by explicit exclusion.
 before_ids={r['id'] for r in existing};folds=[];preds=[]
 for fold,(tr,te) in enumerate(GroupKFold(min(5,len({r['person'] for r in cases})) ).split(cases,groups=[r['person'] for r in cases])):
  before=[cases[j] for j in tr if cases[j]['id'] in before_ids];after=[cases[j] for j in tr];held=[cases[j] for j in te]
  train_people={r['person'] for r in after};held_people={r['person'] for r in held};assert not train_people&held_people
  assert not {s for r in after for s in r['source']}&{s for r in held for s in r['source']}
  bm=fit(before,key);am=fit(after,key);br=fit_rule(before,key,prior_bounds);ar=fit_rule(after,key,prior_bounds)
  for r in held:
   f=r['features'][key];preds.append(dict(id=r['id'],person=r['person'],origin=r['origin'],target=r['levels'][key],fold=fold,before_ai=float(bm.predict([f['x']])[0]),after_ai=float(am.predict([f['x']])[0]),before_rule=rule(f['scalar'],br,higher),after_rule=rule(f['scalar'],ar,higher)))
  folds.append(dict(fold=fold,train_people=sorted(train_people),held_people=sorted(held_people),before_rows=len(before),after_rows=len(after),held_ids=[r['id'] for r in held],before_bounds=br,after_bounds=ar))
 outer={name:metrics(preds,name) for name in ['before_ai','after_ai','before_rule','after_rule']}
 # Fixed historical validation person Mima is never included in either final fit.
 train=[r for r in existing if r['split']=='train'];val=[r for r in existing if r['split']=='validation'];model=fit(train+added,key);bounds=fit_rule(train+added,key,prior_bounds);fixed=[]
 for r in val:
  f=r['features'][key];fixed.append(dict(id=r['id'],person=r['person'],target=r['levels'][key],before_ai=float(baseline['model'].predict([f['x']])[0]),after_ai=float(model.predict([f['x']])[0]),before_rule=rule(f['scalar'],prior_bounds,higher),after_rule=rule(f['scalar'],bounds,higher)))
 fixed_metrics={name:metrics(fixed,name) for name in ['before_ai','after_ai','before_rule','after_rule']}
 ai_adopt=no_regression(outer['after_ai'],outer['before_ai']) and no_regression(fixed_metrics['after_ai'],fixed_metrics['before_ai']);rule_adopt=no_regression(outer['after_rule'],outer['before_rule']) and no_regression(fixed_metrics['after_rule'],fixed_metrics['before_rule'])
 heads[key]=dict(model=model,boundaries=bounds,higher=higher,support=dict(collections.Counter(str(r['levels'][key]) for r in train+added)),approved_ai=ai_adopt,approved_rule=rule_adopt,feature_count=len(train[0]['features'][key]['x']),train_ids=[r['id'] for r in train+added],validation_people=sorted({r['person'] for r in val}))
 reports.append(dict(item=key,existing_rows=len(existing),new_usable_rows=len(added),new_error_rows=0,outer_metrics=outer,outer_predictions=preds,folds=folds,fixed_validation_metrics=fixed_metrics,fixed_validation_predictions=fixed,approved_ai=ai_adopt,approved_rule=rule_adopt,baseline_model='original20' if key=='knee_inward' else 'expanded41'))
 print(key,'normal additions',len(added),'fixed',fixed_metrics,'approved',ai_adopt,rule_adopt,flush=True)
joblib.dump(dict(version='priority-controls-v1',heads=heads,label_sha256=hashlib.sha256(raw).hexdigest(),source_digest=source_sha,expert_validated=False),O/'candidate.joblib')
assert source_sha==digest(read_source());report=dict(complete=True,expert_validated=False,independent_final_test=False,criterion_version='squat-core100-observation-v3',criteria_unchanged=True,heads=reports,return_AI_retrained=False,return_reason='새 복귀 오류가 확인되지 않아 정상-only 모델을 오류 판별 AI로 재학습하지 않음',new_directly_reviewed_sources=64,new_verified_error_sources=0,source_digest=source_sha,labels_sha256=hashlib.sha256(raw).hexdigest(),candidate_sha256=hashlib.sha256((O/'candidate.joblib').read_bytes()).hexdigest(),feature_rows=old+new,limitations=['QEVD는 후보 선정 편향이 있는 개발 대조군; 독립 최종 시험이 아님','정면 무릎 간격 관측 보조항목을 무릎·발 합성 30점으로 치환하지 않음','오류 지지는 기존 시각 초안에서만 확보; 이번 확인에서 새 명확한 오류 없음','원본 학습 모델은 변경하지 않고 후보를 별도 보관'])
(O/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
