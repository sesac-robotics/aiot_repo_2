"""Audited Train-only research candidates; paired fixed-algorithm baselines, no Test tuning."""
from pathlib import Path
import json,sys,collections,hashlib,joblib,numpy as np
sys.dont_write_bytecode=True
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/four-exercise-review-train-v1-20261025/reconciled-v2';P=B/'output/squat-fixed-reference-combined-v3-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'));from split_guard import assert_training_allowed
new=json.loads((O/'score-inputs.frozen.json').read_text());old=[r for r in json.loads((P/'score-inputs.frozen.json').read_text()) if r['split']=='train' and r['exercise']!='squats'];assert_training_allowed(new);assert_training_allowed(old);assert len(new)==len(old)==609
# Union all known person/source/previous connected-group identities before any fold.
par={r['id']:r['id'] for r in new}
def root(x):
 while par[x]!=x:par[x]=par[par[x]];x=par[x]
 return x
def join(a,b):par[root(a)]=root(b)
seen={}
for r in new:
 for k in ['person:'+r['person'],'parent_group:'+str(r['group'])]+['sha:'+h for h in r['sources']]:
  if k in seen:join(r['id'],seen[k])
  else:seen[k]=r['id']
group={r['id']:root(r['id']) for r in new};heads=sorted({(r['exercise'],x['item'],x['bank']) for r in new for x in r['items']})
def eligible(rows,key):
 out=[]
 for r in rows:
  if r['exercise']!=key[0]:continue
  for item,y in zip(r['items'],r['levels']):
   if (item['item'],item['bank'])!=key[1:] or y is None or not item['eligible'] or item.get('feature') is None:continue
   x=np.array(item['feature'],float)
   if np.isfinite(x).all():out.append(dict(id=r['id'],group=group[r['id']],person=r['person'],sources=r['sources'],x=x,y=int(y)))
 return out
def support(a):return {str(g):dict(rows=sum(r['y']==g for r in a),groups=len({r['group'] for r in a if r['y']==g})) for g in [0,1,2]}
def admitted(a):
 s=support(a);valid={int(g) for g,v in s.items() if v['groups']>=2};return [r for r in a if r['y'] in valid]
def can_fit(a):return len(a)>=8 and len({r['group'] for r in a if r['y']==2})>=2 and len({r['group'] for r in a if r['y']<2})>=2
ALGS=['balanced_logistic_C1','ExtraTrees64_minleaf3']
def fit(a,alg):
 assert can_fit(a);pairs=collections.Counter((r['y'],r['group']) for r in a);gg=collections.Counter(y for y,g in pairs);w=np.array([1/(gg[r['y']]*pairs[r['y'],r['group']]) for r in a]);w=w/w.mean();X=np.array([r['x'] for r in a]);y=np.array([r['y'] for r in a])
 if alg==ALGS[0]:m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=251025));m.fit(X,y,logisticregression__sample_weight=w)
 else:m=ExtraTreesClassifier(n_estimators=64,min_samples_leaf=3,random_state=251025,n_jobs=1);m.fit(X,y,sample_weight=w)
 return m
def metric(pred):
 ys=[r['target'] for r in pred];done=[r for r in pred if r['prediction'] is not None];e=[r for r in done if r['target']<2];n=[r for r in done if r['target']==2];return dict(eligible=len(pred),evaluated=len(done),abstained=len(pred)-len(done),errors_detected=sum(r['prediction']<2 for r in e),observed_errors=len(e),all_eligible_errors=sum(y<2 for y in ys),normal_false_alarms=sum(r['prediction']<2 for r in n),observed_normals=len(n),all_eligible_normals=sum(y==2 for y in ys),exact_grade_correct=sum(r['prediction']==r['target'] for r in done),exact_grade_total=len(done),error_recall=sum(r['prediction']<2 for r in e)/len(e) if e else None,normal_false_alarm_rate=sum(r['prediction']<2 for r in n)/len(n) if n else None)
reports=[];models={};fold_receipts=[]
for key in heads:
 a=eligible(new,key);b=eligible(old,key);entry=dict(exercise=key[0],item=key[1],bank=key[2],candidate_eligible_rows=len(a),candidate_support=support(a),parent_support=support(b),algorithms=[]);groups=sorted({r['group'] for r in a});
 if len(groups)<4:entry['status']='unsupported_fewer_than_four_known_groups';reports.append(entry);continue
 sizes=collections.Counter(len(r['x']) for r in a);assert len(sizes)==1,(key,sizes)
 splits=list(GroupKFold(4).split(np.zeros(len(a)),groups=[r['group'] for r in a]));foldmap={r['id']:fold for fold,(_,va) in enumerate(splits) for r in [a[i] for i in va]}
 for alg in ALGS:
  preds={'audited':[],'old_labels_baseline':[]};folds=[]
  for fold,(tr,va) in enumerate(splits):
   held={a[i]['group'] for i in va};test=[a[i] for i in va];f=dict(fold=fold,held_groups=sorted(held),validation_ids=[r['id'] for r in test])
   for version,allrows in [('audited',a),('old_labels_baseline',b)]:
    fitrows=admitted([r for r in allrows if r['group'] not in held]);assert not held&{r['group'] for r in fitrows};f[version]=dict(train_ids=[r['id'] for r in fitrows],train_groups=sorted({r['group'] for r in fitrows}),support=support(fitrows),status='trained' if can_fit(fitrows) else 'abstain_insufficient_grade_group_support')
    m=fit(fitrows,alg) if can_fit(fitrows) else None;pp=m.predict(np.array([r['x'] for r in test])).tolist() if m is not None else [None]*len(test)
    preds[version].extend(dict(id=r['id'],group=r['group'],fold=fold,target=r['y'],prediction=int(p) if p is not None else None) for r,p in zip(test,pp))
   folds.append(f)
  full=admitted(a);keystr='|'.join(key)+'|'+alg;trained=can_fit(full)
  if trained:models[keystr]=dict(model=fit(full,alg),trained_ids=[r['id'] for r in full],train_groups=sorted({r['group'] for r in full}),grade_support=support(full),exercise=key[0],item=key[1],bank=key[2],grade_values=sorted({r['y'] for r in full}),status='research_only_no_live_promotion')
  entry['algorithms'].append(dict(name=alg,status='trained_research' if trained else 'unsupported_normal_error_group_support',actual_fit_ids=[r['id'] for r in full] if trained else [],excluded_sparse_grade_ids=[r['id'] for r in a if r['id'] not in {z['id'] for z in full}],candidate_cv=metric(preds['audited']),old_labels_baseline_cv_same_audited_targets=metric(preds['old_labels_baseline']),predictions=preds,folds=folds))
 entry['status']='evaluated_fixed_candidates';reports.append(entry);(O/'training-progress.json').write_text(json.dumps(dict(completed_heads=len(reports),total_heads=len(heads),last=key),indent=2)+'\n');print(key,len(a),[x['status'] for x in entry['algorithms']],flush=True)
# The entire row list is not the fit list. Persist exact task-level IDs and all omissions.
joblib.dump(dict(schema='four-exercise-audited-score-candidates-v1',models=models,inputs_sha256=hashlib.sha256((O/'score-inputs.frozen.json').read_bytes()).hexdigest(),scope='four_exercises_research_only',live_deployed=False),O/'score-candidates.joblib')
summary={}
for ex in ['shoulder','pushups','ytw','plank']:
 hs=[h for h in reports if h['exercise']==ex];actual={i for h in hs for al in h.get('algorithms',[]) for i in al['actual_fit_ids']};summary[ex]=dict(candidate_rows=sum(r['exercise']==ex for r in new),actual_fit_unique_rows=len(actual),actual_fit_ids=sorted(actual),unfitted_ids=sorted(r['id'] for r in new if r['exercise']==ex and r['id'] not in actual),trained_head_algorithms=sum(al['status']=='trained_research' for h in hs for al in h.get('algorithms',[])),unsupported_banks=[dict(item=h['item'],bank=h['bank'],support=h['candidate_support']) for h in hs if not any(a['status']=='trained_research' for a in h.get('algorithms',[]))])
report=dict(protocol='Train-only fixed four-fold known-person/source-connected paired label audit comparison',baseline='same two fixed algorithms refitted using previous labels; not_parent_deployed_model_comparison',model_selection='both_fixed_candidates_retained_no_Test_based_selection',scope='exposed_Train_CV_diagnostic_not_independent_final_Test',manual_labels='AI_visual_draft_not_expert_confirmed',squats_preserved=True,live_deployed=False,summary=summary,heads=reports)
(O/'score-training.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print({k:{a:v[a] for a in ['candidate_rows','actual_fit_unique_rows','trained_head_algorithms']} for k,v in summary.items()})
