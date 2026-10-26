"""Isolated fixed research fits with person/source-group validation and abstention."""
from pathlib import Path
import os,json,sys,hashlib,collections
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
if hasattr(os,'sched_getaffinity'):os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
import numpy as np,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-expanded-train-v3-20261025';sys.path.insert(0,str(B/'tools/error-enrichment'));from split_guard import assert_training_allowed
rows=json.loads((O/'training-inputs.frozen.json').read_text());assert_training_allowed(rows)
for r in rows:assert r['visual_qa']['all_actual_context_frames_reviewed'];assert len(r['features'])==6;assert np.isfinite(r['features']).all()
models={};reports={}
def make(kind):
 if kind=='body_logistic':return make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',solver='liblinear',random_state=20261025))
 if kind=='lower_svm':return make_pipeline(StandardScaler(),OneClassSVM(nu=.1,gamma='scale'))
 return make_pipeline(StandardScaler(),IsolationForest(n_estimators=128,contamination=.1,random_state=20261025,n_jobs=1))
for kind,item in [('body_logistic','frontal_trunk_hold'),('lower_svm','lower_return'),('lower_isolation_forest','lower_return')]:
 rr=[r for r in rows if r['item']==item];people=sorted({r['person'] for r in rr});assert len(people)==5;assert len({r['feature_schema'] for r in rr})==1
 m=make(kind).fit([r['features'] for r in rr],[r['grade'] for r in rr]) if kind=='body_logistic' else make(kind).fit([r['features'] for r in rr]);cv=[]
 for person in people:
  tr=[r for r in rr if r['person']!=person];te=[r for r in rr if r['person']==person];assert not {r['source_sha256'] for r in tr}&{r['source_sha256'] for r in te};train_people=sorted({r['person'] for r in tr})
  if kind=='body_logistic' and {r['grade'] for r in tr}!={1,2}:
   cv.extend(dict(id=r['id'],person=person,truth=r['grade'],prediction=None,status='abstained_training_fold_missing_error_class',training_people=train_people) for r in te);continue
  fold=make(kind).fit([r['features'] for r in tr],[r['grade'] for r in tr]) if kind=='body_logistic' else make(kind).fit([r['features'] for r in tr]);pred=fold.predict([r['features'] for r in te])
  cv.extend(dict(id=r['id'],person=person,truth=r['grade'],prediction=int(v),status='research_grade_prediction' if kind=='body_logistic' else 'normal_reference_inlier_indicator_not_grade',training_people=train_people) for r,v in zip(te,pred))
 report=dict(item=item,rows=len(rr),people=people,grades=dict(collections.Counter(r['grade'] for r in rr)),leave_person_out=cv,source_overlap=False,production_eligible=False,independent_final_test=False)
 if kind=='body_logistic':
  observed=[r for r in cv if r['prediction'] is not None];errors=[r for r in cv if r['truth']==1];report.update(covered_rows=len(observed),all_rows=len(cv),covered_correct=sum(r['truth']==r['prediction'] for r in observed),normal_false_error=sum(r['truth']==2 and r['prediction']==1 for r in observed),normal_covered=sum(r['truth']==2 for r in observed),error_rows=len(errors),error_rows_with_valid_multiclass_fold=sum(r['prediction'] is not None for r in errors),independent_error_recall=None,grade0_supported=False,adoption_blockers=['only_one_error_person','all_error_rows_abstained_in_person_excluded_validation','no_independent_final_test','severe_grade0_untrained','no_live_event_detector_validation'])
 else:report.update(normal_inliers=sum(r['prediction']==1 for r in cv),normal_rows=len(cv),error_detection_accuracy=None,grade_prediction_enabled=False)
 models[kind]=dict(model=m,item=item,posture='seated',view='front',feature_schema=rr[0]['feature_schema'],criterion=rr[0]['criterion'],feature_names=rr[0]['feature_names'],training_ids=[r['id'] for r in rr],training_people=people,semantics='research_grade1_2_only_not_points' if kind=='body_logistic' else 'normal_reference_not_grade',grade_prediction_enabled=False,score_prediction_enabled=False,production_eligible=False);reports[kind]=report
artifact=dict(schema='shoulder-expanded-Train-research-v3',heads=models,training_inputs_sha256=hashlib.sha256((O/'training-inputs.frozen.json').read_bytes()).hexdigest(),grade_prediction_enabled=False,score_prediction_enabled=False,production_eligible=False,standing_models_preserved_at=str(B/'output/shoulder-rebuild-v2-20261025/normal-reference-rebuilt.joblib'))
p=O/'research-models.joblib';assert not p.exists();joblib.dump(artifact,p)
report=dict(reports=reports,selection='fixed hyperparameters; no model winner selected or deployed',model_fits=3,movement_grade_research_fits=1,normal_reference_fits=2,validation_type='Train_leave_person_out_with_same_source_group_exclusion',Test_inference=False,independent_final_test=False,deployment_changed=False,severe_grade0_supported=False)
(O/'model-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:{a:b for a,b in r.items() if a!='leave_person_out'} for k,r in reports.items()},ensure_ascii=False,indent=2))
