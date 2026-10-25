"""Connected-person nested development CV; no pretend untouched final test."""
import sys,json,collections,hashlib,copy
from pathlib import Path
import joblib,numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent;sys.path.insert(0,str(R/'output/live-webcam-20261023'))
from runtime_snapshot import prepare_snapshot,read_source,digest
runtime,source_sha=prepare_snapshot();sys.path.insert(0,str(runtime));from app.scoring.rules import rule_grade
raw=(O/'labels.aligned.frozen.json').read_bytes();rows=json.loads(raw)['rows'];ids=['descent_range','return_extension','knee_foot','trunk_balance'];prior=joblib.load(runtime/'artifacts/posture_score/squats.joblib')
# Connected components cover person aliases, whole source hashes, and recording groups.
parent=list(range(len(rows)));seen={}
def root(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
for j,r in enumerate(rows):
 person=r['person'].replace('rehab_','rehab26_')
 tokens=['person:'+person]+['sha:'+s for s in r['sources']]+['rec:'+s for s in r.get('recording_groups',[])]
 # REHAB24 vs REHAB26 physical identity linkage is unresolved: conservatively hold
 # the two families together, rather than claiming cross-family unseen people.
 if person.startswith(('rehab24_','rehab26_')):tokens.append('unresolved_cross_REHAB_identity_family')
 for t in tokens:
  if t in seen:parent[root(j)]=root(seen[t])
  else:seen[t]=j
for j,r in enumerate(rows):r['connected_person_group']='g'+str(root(j))
(O/'partition.json').write_text(json.dumps(dict(rows=[dict(id=r['id'],person=r['person'],group=r['connected_person_group'],sources=r['sources']) for r in rows],groups=len({r['connected_person_group'] for r in rows}),unresolved_REHAB_families_held_together=True,independent_final_test=False),ensure_ascii=False,indent=2))

def fit(data,indices,kind):
 x=np.asarray([data[j]['x'] for j in indices]);y=np.asarray([data[j]['target'] for j in indices]);groups=[data[j]['group'] for j in indices]
 counts=collections.Counter((g,t) for g,t in zip(groups,y));levels=collections.Counter(y)
 if kind=='prior_recipe':w=np.ones(len(y));leaf=2
 elif kind=='group_balanced':w=np.array([1/counts[g,t] for g,t in zip(groups,y)]);leaf=3
 else:w=np.array([1/levels[t] for t in y]);leaf=2
 if kind!='prior_recipe' and x.shape[1]==13:x[:,8:11]=0 # exclude absolute foot/tibia pitch as automatic error proxies
 model=ExtraTreesRegressor(n_estimators=80,min_samples_leaf=leaf,random_state=23,n_jobs=2).fit(x,y,sample_weight=w)
 return model

def classify(v,b,higher):
 good,partial=b[:2];return 2 if (v>=good if higher else v<=good) else 1 if (v>=partial if higher else v<=partial) else 0

def tune_rule(data,indices,prior_bounds,higher):
 # Only training scalar/heel signals, never model predictions or held labels.
 vals=sorted({data[j]['scalar'] for j in indices});cuts=[vals[0]-.01]+[(a+b)/2 for a,b in zip(vals,vals[1:])]+[vals[-1]+.01]
 if len(cuts)>25:cuts=list(np.quantile(cuts,np.linspace(0,1,25)))
 choices=[tuple(prior_bounds[:2])]
 for good in cuts:
  for partial in cuts:
   if (higher and good>partial) or (not higher and good<partial):choices.append((good,partial))
 counts=collections.Counter((data[j]['group'],data[j]['target']) for j in indices)
 def loss(b):return np.average([abs(rule_value(data[j],b,higher)-data[j]['target']) for j in indices],weights=[1/counts[data[j]['group'],data[j]['target']] for j in indices])
 return min(choices,key=lambda b:(loss(b),tuple(b)!=tuple(prior_bounds[:2])))
def rule_value(r,b,higher):
 g=classify(r['scalar'],b,higher)
 if r['heel'] is not None:g=min(g,2 if r['heel']<=.1 else 1 if r['heel']<=.2 else 0)
 return g

def original_rule(r,head,higher):
 if head.get('boundary') is None:return None
 note=dict(eligible=True,measurement=r['scalar'],feature=r['x'],higher_is_better=higher,item='knee_foot' if r['heel'] is not None else '')
 value=rule_grade(note,head['boundary'])
 if head.get('explicit_rule_kind')=='knee_heel_pitch_v1' and head.get('pitch_boundary') is not None:
  good,partial=head['pitch_boundary'];v=r['x'][9];value=min(value,2 if v<=good else 1 if v<=partial else 0)
 return value

def metrics(preds,field):
 a=[p for p in preds if p.get(field) is not None];normal=[p for p in a if p['target']==2];bad=[p for p in a if p['target']<2]
 return dict(n=len(a),MAE_grade=float(np.mean([abs(p[field]-p['target']) for p in a])) if a else None,error_MAE=float(np.mean([abs(p[field]-p['target']) for p in bad])) if bad else None,error_n=len(bad),false_warning_normal=float(np.mean([p[field]<1.5 for p in normal])) if normal else None,missed_error=float(np.mean([p[field]>=1.5 for p in bad])) if bad else None)
report=[];models={}
for i,key in enumerate(ids):
 banks=sorted({r['items'][i]['bank'] for r in rows if r['items'][i]['eligible'] and r['levels'][i] is not None})
 for bank in banks:
  h=prior.get(('squats',key,bank));data=[]
  for r in rows:
   n=r['items'][i]
   if not n['eligible'] or r['levels'][i] is None or n['bank']!=bank:continue
   x=n['feature']
   if not np.isfinite(x).all():continue
   # Native notes have measurement; older cache rows store the 90th/10th percentile.
   scalar=n.get('measurement',x[1] if i==0 else x[5]);higher=i==1
   data.append(dict(id=r['id'],person=r['person'],group=r['connected_person_group'],x=x,target=r['levels'][i],scalar=scalar,heel=x[-1] if i==2 else None))
  support=dict(collections.Counter(str(r['target']) for r in data));groups={r['group'] for r in data};error_groups={r['group'] for r in data if r['target']<2}
  entry=dict(item=key,bank=bank,n=len(data),groups=len(groups),support=support,error_groups=len(error_groups),weight=[30,20,30,20][i])
  if len(set(r['target'] for r in data))<2 or len(groups)<3 or h is None:
   entry.update(trained=False,reason='단일 정상 라벨 또는 사람 그룹/동일 구도 기존 모델 부족',promoted_ai=False,promoted_rule=False);report.append(entry);continue
  preds=[];folds=[];b0=h.get('boundary') or ((160,140) if higher else (8,15) if i==3 else (.3,.8))
  for fold,(tr,te) in enumerate(GroupKFold(min(5,len(groups))).split(data,groups=[r['group'] for r in data])):
   assert not {data[j]['group'] for j in tr}&{data[j]['group'] for j in te}
   # Model recipe choice is nested, training groups only.
   candidates=['prior_recipe','group_balanced','class_balanced'];scores={}
   if len({data[j]['group'] for j in tr})>=3:
    local=[data[j] for j in tr]
    for kind in candidates:
     errors=[]
     for a,b in GroupKFold(min(3,len({r['group'] for r in local}))).split(local,groups=[r['group'] for r in local]):
      m=fit(local,a,kind);q=m.predict([local[j]['x'] for j in b]);errors.extend(abs(q-np.array([local[j]['target'] for j in b])))
     scores[kind]=float(np.mean(errors))
    chosen=min(candidates,key=lambda c:(scores[c],c!='prior_recipe'))
   else:chosen='prior_recipe'
   before=fit(data,tr,'prior_recipe');after=fit(data,tr,chosen);bounds=tune_rule(data,tr,b0,higher)
   for j in te:
    r=data[j];old_raw=float(h['model'].predict([r['x']])[0]) if not isinstance(h['model'],dict) else None
    preds.append(dict(id=r['id'],group=r['group'],target=r['target'],fold=fold,prior_recipe_ai=float(before.predict([r['x']])[0]),after_ai=float(after.predict([r['x']])[0]),prior_rule=original_rule(r,h,higher),after_rule=rule_value(r,bounds,higher),frozen_old_ai_replay=old_raw))
   folds.append(dict(fold=fold,train_groups=sorted({data[j]['group'] for j in tr}),held_groups=sorted({data[j]['group'] for j in te}),selected_recipe=chosen,inner_MAE=scores,rule_bounds=list(bounds)))
  met={k:metrics(preds,k) for k in ['prior_recipe_ai','after_ai','prior_rule','after_rule','frozen_old_ai_replay']}
  # Keep uncertainty explicit: development adoption only if normal & error performance
  # do not regress; error support must span at least two connected groups.
  def better(a,b):return b['MAE_grade'] is not None and a['MAE_grade'] is not None and a['MAE_grade']<b['MAE_grade']-1e-8 and a['error_MAE'] is not None and a['error_MAE']<=b['error_MAE']+1e-8 and a['false_warning_normal']<=b['false_warning_normal']+1e-8
  adopt_ai=len(error_groups)>=2 and better(met['after_ai'],met['prior_recipe_ai']);adopt_rule=len(error_groups)>=2 and better(met['after_rule'],met['prior_rule'])
  # Fit final development candidate on the old explicit train groups only; validation
  # groups are not silently added to production. Overlapping components excluded.
  held={r['connected_person_group'] for r in rows if r['split']=='validation'}
  trfinal=[j for j,r in enumerate(data) if r['group'] not in held]
  if len(trfinal)<3:adopt_ai=adopt_rule=False;trfinal=list(range(len(data)))
  final_error_groups={data[j]['group'] for j in trfinal if data[j]['target']<2}
  if len(final_error_groups)<2:adopt_ai=adopt_rule=False
  finalkind=collections.Counter(f['selected_recipe'] for f in folds).most_common(1)[0][0]
  m=fit(data,trfinal,finalkind);b=tune_rule(data,trfinal,b0,higher)
  models[(key,bank)]=dict(model=m,boundaries=list(b),higher=higher,support=support,adopt_ai=adopt_ai,adopt_rule=adopt_rule,fit_role='development_only',feature_dimension=len(data[0]['x']),train_ids=[data[j]['id'] for j in trfinal],train_groups=sorted({data[j]['group'] for j in trfinal}))
  entry.update(trained=True,promoted_ai=adopt_ai,promoted_rule=adopt_rule,metrics=met,folds=folds,predictions=preds,final_recipe=finalkind,final_fit_rows=len(trfinal));report.append(entry)
  print(key,bank,met['prior_recipe_ai']['MAE_grade'],met['after_ai']['MAE_grade'],adopt_ai,adopt_rule,flush=True)
joblib.dump(dict(version='squat-unified-core100-v3',rubric_sha256=hashlib.sha256((O/'rubric.json').read_bytes()).hexdigest(),labels_sha256=hashlib.sha256(raw).hexdigest(),heads=models,source_digest=source_sha),O/'candidate.joblib')
assert source_sha==digest(read_source())
(O/'comparison.json').write_text(json.dumps(dict(version='squat-unified-core100-v3',heads=report,source_preserved=True,source_digest=source_sha,independent_final_test=False,validation='nested connected-person development CV; frozen deployed predictions may overlap original training and are replay-only',rubric_sha256=hashlib.sha256((O/'rubric.json').read_bytes()).hexdigest(),labels_sha256=hashlib.sha256(raw).hexdigest(),candidate_sha256=hashlib.sha256((O/'candidate.joblib').read_bytes()).hexdigest()),ensure_ascii=False,indent=2))
