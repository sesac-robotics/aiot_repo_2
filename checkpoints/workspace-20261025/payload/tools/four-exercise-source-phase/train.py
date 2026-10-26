"""Fixed candidates, identical connected-group folds and audited targets; no Test tuning."""
from pathlib import Path
import json,sys,collections,hashlib,joblib,numpy as np
sys.dont_write_bytecode=True
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/four-exercise-source-phase-v2-20261025';P=B/'output/four-exercise-review-train-v1-20261025/reconciled-v2'
sys.path.insert(0,str(B/'tools/error-enrichment'));from split_guard import assert_training_allowed
new=json.loads((O/'score-inputs.frozen.json').read_text());old=json.loads((O/'paired-baseline-inputs.frozen.json').read_text());parent={r['id']:r for r in json.loads((P/'score-inputs.frozen.json').read_text())};assert_training_allowed(new);assert_training_allowed(old)
# Restore parent eligibility for virtual baseline banks. Revised observation
# holds are part of the candidate and must not silently alter its baseline.
for r in old:
 p=parent.get(r['id'])
 if p is None:continue
 for x in r['items']:
  if not x.get('paired_baseline_only'):continue
  item=x['item'];bank=x['bank'];wanted=(('upright' if bank.startswith('upright_') else 'hinged')+'_compact_shape_v3') if '_stage_contiguous_v4' in bank else None
  if bank=='observed_post_descent_return_v2':
   original=next((z for z in p['items'] if z['item']==item and z['bank'] in ('front','side_legacy_roi_v1') and z.get('feature') is not None and len(z['feature'])==8),None)
  else:original=next((z for z in p['items'] if z['item']==item and (wanted is None or z['bank']==wanted)),None)
  if original is not None:
   x.clear();x.update(original);x.update(bank=bank,paired_baseline_only=True)
  else:x.update(feature=None,eligible=False,holds=['no_unit_equivalent_original_feature_contract'])
(O/'paired-baseline-used-v2.frozen.json').write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n')
par={r['id']:r['id'] for r in new}
def root(x):
 while par[x]!=x:par[x]=par[par[x]];x=par[x]
 return x
seen={}
for r in new:
 for k in ['person:'+r['person'],'parent_group:'+str(r['group'])]+['sha:'+s for s in r['sources']]:
  if k in seen:par[root(r['id'])]=root(seen[k])
  else:seen[k]=r['id']
groups={r['id']:root(r['id']) for r in new}
splits=list(GroupKFold(4).split(np.zeros(len(new)),groups=[groups[r['id']] for r in new]));foldmap={new[i]['id']:fold for fold,(_,va) in enumerate(splits) for i in va}
foldtext=json.dumps(dict(group_by_id=groups,fold_by_id=foldmap,policy='all_known_people_original_source_SHA_and_parent_groups_connected_before_any_head_fit; overlapping_or_synchronized_views_together'),indent=2)+'\n'
if (O/'folds.frozen.json').exists():assert (O/'folds.frozen.json').read_text()==foldtext
else:(O/'folds.frozen.json').write_text(foldtext)
heads=sorted({(r['exercise'],x['item'],x['bank']) for r in new for x in r['items']})
def records(rows,key):
 out={}
 for r in rows:
  if r['exercise']!=key[0]:continue
  for it,y in zip(r['items'],r['levels']):
   if (it['item'],it['bank'])!=key[1:]:continue
   x=np.array(it['feature'],float) if it.get('feature') is not None else None
   ok=it['eligible'] and x is not None and np.isfinite(x).all()
   out[r['id']]=dict(id=r['id'],group=groups[r['id']],person=r['person'],sources=r['sources'],x=x if ok else None,y=y,holds=it.get('holds',[]))
 return out
def support(a):return {str(g):dict(rows=sum(r['y']==g for r in a),groups=len({r['group'] for r in a if r['y']==g})) for g in [0,1,2]}
def admitted(a):
 good=[r for r in a if r['y'] is not None and r['x'] is not None];s=support(good);valid={int(g) for g,v in s.items() if v['groups']>=2};return [r for r in good if r['y'] in valid]
def canfit(a):return len(a)>=8 and len({r['group'] for r in a if r['y']==2})>=2 and len({r['group'] for r in a if r['y']<2})>=2
ALGS=['balanced_logistic_C1','ExtraTrees64_minleaf3']
def fit(a,alg):
 pairs=collections.Counter((r['y'],r['group']) for r in a);gg=collections.Counter(y for y,g in pairs);w=np.array([1/(gg[r['y']]*pairs[r['y'],r['group']]) for r in a]);w/=w.mean();X=np.array([r['x'] for r in a]);y=np.array([r['y'] for r in a],int)
 if alg==ALGS[0]:m=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=251025));m.fit(X,y,logisticregression__sample_weight=w)
 else:m=ExtraTreesClassifier(n_estimators=64,min_samples_leaf=3,random_state=251025,n_jobs=1);m.fit(X,y,sample_weight=w)
 return m
def metric(pred):
 done=[r for r in pred if r['prediction'] is not None];e=[r for r in done if r['target']<2];n=[r for r in done if r['target']==2];allerr=[r for r in pred if r['target']<2]
 return dict(labelled_targets=len(pred),evaluated=len(done),abstained=len(pred)-len(done),coverage=len(done)/len(pred) if pred else 0,errors_detected=sum(r['prediction']<2 for r in e),observed_errors=len(e),all_labelled_errors=len(allerr),normal_false_alarms=sum(r['prediction']<2 for r in n),observed_normals=len(n),all_labelled_normals=sum(r['target']==2 for r in pred),error_recall=sum(r['prediction']<2 for r in e)/len(e) if e else None,end_to_end_error_recall=sum(r['prediction']<2 for r in e)/len(allerr) if allerr else None,normal_false_alarm_rate=sum(r['prediction']<2 for r in n)/len(n) if n else None,exact_grade_correct=sum(r['prediction']==r['target'] for r in done),exact_grade_total=len(done))
reports=[];models={}
for key in heads:
 a=records(new,key);b=records(old,key);targets=[r for r in a.values() if r['y'] is not None];entry=dict(exercise=key[0],item=key[1],bank=key[2],labelled_targets=len(targets),candidate_observable=sum(r['x'] is not None for r in targets),candidate_support=support([r for r in targets if r['x'] is not None]),algorithms=[])
 for alg in ALGS:
  preds={'candidate':[],'baseline':[]};fr=[]
  for fold in range(4):
   val=[r for r in targets if foldmap[r['id']]==fold];held={groups[r['id']] for r in new if foldmap[r['id']]==fold}
   receipt=dict(fold=fold,held_groups=sorted(held))
   for name,allrows in [('candidate',a),('baseline',b)]:
    tr=admitted([r for r in allrows.values() if foldmap[r['id']]!=fold]);assert not {r['group'] for r in tr}&held
    m=fit(tr,alg) if canfit(tr) else None;receipt[name]=dict(status='trained' if m is not None else 'abstain_insufficient_grade_group_support',train_ids=[r['id'] for r in tr] if m is not None else [],grade_support=support(tr))
    for target in val:
     observed=allrows.get(target['id']);x=observed['x'] if observed is not None else None
     p=int(m.predict(np.array([x]))[0]) if m is not None and x is not None else None
     reason=None if p is not None else 'insufficient_train_support' if m is None else 'required_observation_unavailable'
     preds[name].append(dict(id=target['id'],group=target['group'],fold=fold,target=int(target['y']),prediction=p,abstention_reason=reason))
   fr.append(receipt)
  full=admitted(list(a.values()));trained=canfit(full);ss=support(full);mm=metric(preds['candidate']);errors={r['group'] for r in full if r['y']<2};normal={r['group'] for r in full if r['y']==2};fail=[]
  if min(len(errors),len(normal))<3:fail.append('normal_or_error_fewer_than_three_connected_groups')
  if mm['error_recall'] is None or mm['error_recall']<.8:fail.append('error_recall_below_0.80_or_unavailable')
  if mm['normal_false_alarm_rate'] is None or mm['normal_false_alarm_rate']>.15:fail.append('normal_false_alarm_above_0.15_or_unavailable')
  if mm['coverage']<.8:fail.append('labelled_target_coverage_below_0.80')
  if not trained:fail.append('cannot_fit_supported_normal_error_classifier')
  sparse=[int(g) for g,s in entry['candidate_support'].items() if s['rows'] and s['groups']<2]
  if sparse:fail.append('observed_grade_has_fewer_than_two_training_groups')
  if trained:models['|'.join(key)+'|'+alg]=dict(model=fit(full,alg),trained_ids=[r['id'] for r in full],train_groups=sorted({r['group'] for r in full}),grade_support=ss,supported_grades=sorted({r['y'] for r in full}),status='research_only',independent_final_Test_passed=False,live_deployed=False)
  common={r['id'] for r in preds['candidate'] if r['prediction'] is not None}&{r['id'] for r in preds['baseline'] if r['prediction'] is not None}
  entry['algorithms'].append(dict(name=alg,status='trained_research' if trained else 'unsupported',actual_fit_ids=[r['id'] for r in full] if trained else [],candidate_cv=mm,baseline_cv_same_targets=metric(preds['baseline']),paired_common_observed=dict(candidate=metric([r for r in preds['candidate'] if r['id'] in common]),baseline=metric([r for r in preds['baseline'] if r['id'] in common])),development_gate_passed=not fail,gate_failures=fail,unsupported_sparse_grades=sparse,predictions=preds,folds=fr))
 reports.append(entry);(O/'training-progress.json').write_text(json.dumps(dict(completed_heads=len(reports),total_heads=len(heads),last=key),indent=2)+'\n');print(key,entry['candidate_observable'],[z['status'] for z in entry['algorithms']],flush=True)
joblib.dump(dict(schema='four-exercise-source-phase-v2',models=models,inputs_sha256=hashlib.sha256((O/'score-inputs.frozen.json').read_bytes()).hexdigest(),scope='Train_only_research',live_deployed=False),O/'score-candidates.joblib')
summary={}
for ex in ['shoulder','pushups','ytw','plank']:
 hs=[h for h in reports if h['exercise']==ex];ids={i for h in hs for a in h['algorithms'] for i in a['actual_fit_ids']};summary[ex]=dict(candidate_records=sum(r['exercise']==ex for r in new),actual_fit_unique_records=len(ids),actual_fit_ids=sorted(ids),unfitted_ids=sorted(r['id'] for r in new if r['exercise']==ex and r['id'] not in ids),trained_head_algorithms=sum(a['status']=='trained_research' for h in hs for a in h['algorithms']),development_gate_passes=[dict(item=h['item'],bank=h['bank'],algorithm=a['name']) for h in hs for a in h['algorithms'] if a['development_gate_passed']])
report=dict(protocol='fixed_four_fold_all_known_person_source_parent_group_connected_Train_CV',comparison='original_and_revised_features_evaluated_on_identical_audited_target_intervals; each_model_receives_its_own_feature_contract',coverage_denominator='all_labelled_targets_in_head_including_observation_holds_and_unfit_folds',scope='exposed_Train_development_diagnostic_not_independent_final_Test',fixed_algorithms=ALGS,model_selection='retain_both_no_Test_based_selection',manual_labels='AI_source_visual_draft_not_independent_expert',squats_preserved=True,live_deployed=False,summary=summary,heads=reports)
(O/'score-training.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print({k:{z:v[z] for z in ['candidate_records','actual_fit_unique_records','trained_head_algorithms']} for k,v in summary.items()})
