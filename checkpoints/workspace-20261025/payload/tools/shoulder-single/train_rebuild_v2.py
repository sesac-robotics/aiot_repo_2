"""Fit isolated normal-reference heads; a lack of error labels blocks grade fitting."""
from pathlib import Path
import os,json,sys,collections,hashlib
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-rebuild-v2-20261025';sys.path.insert(0,str(B/'tools/error-enrichment'));from split_guard import assert_training_allowed
rows=json.loads((O/'training-inputs.frozen.json').read_text());assert_training_allowed(rows)
# Canonical source/person components cannot cross the leave-person-out folds.
people_by_sha=collections.defaultdict(set)
for r in rows:people_by_sha[r['source_sha256']].add(r['person'])
assert all(len(v)==1 for v in people_by_sha.values())
def make(kind):
 if kind=='svm':return make_pipeline(StandardScaler(),OneClassSVM(nu=.1,gamma='scale'))
 return make_pipeline(StandardScaler(),IsolationForest(n_estimators=128,contamination=.1,random_state=20261025,n_jobs=1))
heads={};reports={};capabilities=[]
for key in sorted({(r['item'],r['posture']) for r in rows}):
 rr=[r for r in rows if (r['item'],r['posture'])==key];people=sorted({r['person'] for r in rr});grades=dict(collections.Counter(str(r['grade']) for r in rr));name='/'.join(key)
 capabilities.append({'item':key[0],'posture':key[1],'grade_counts':grades,'grade_prediction_enabled':False,'points':None,'blocked_reason':'normal_only_no_observed_error_grades','training_ids':[r['id'] for r in rr]})
 if len(people)<3:
  reports[name]={'status':'held_reference_fit','reason':'fewer_than_three_Train_people','rows':len(rr),'people':people};continue
 models={};comparison={}
 for kind in ['svm','isolation_forest']:
  m=make(kind).fit([r['features'] for r in rr]);models[kind]=m;cv=[]
  for person in people:
   tr=[r for r in rr if r['person']!=person];te=[r for r in rr if r['person']==person]
   assert not {r['source_sha256'] for r in tr}&{r['source_sha256'] for r in te}
   fold=make(kind).fit([r['features'] for r in tr]);pred=fold.predict([r['features'] for r in te])
   cv.extend([{'id':r['id'],'person':person,'normal_reference_inlier':bool(v==1),'training_people':sorted({x['person'] for x in tr})} for r,v in zip(te,pred)])
  comparison[kind]={'normal_inliers':sum(x['normal_reference_inlier'] for x in cv),'normal_rows':len(cv),'leave_person_out':cv,'error_examples':0,'error_detection_accuracy':None}
 # Preserve both fixed research fits. No winner is approved from normal-only metrics.
 heads[key]={'models':models,'criterion':rr[0]['criterion'],'feature_schema':rr[0]['feature_schema'],'feature_names':rr[0]['feature_names'],'training_ids':[r['id'] for r in rr],'training_source_hashes':sorted({r['source_sha256'] for r in rr}),'selection_for_grade_or_deployment':None}
 reports[name]={'status':'normal_reference_fitted','rows':len(rr),'people':people,'comparison':comparison,'grade_prediction_enabled':False}
artifact={'schema':'shoulder-aligned-normal-reference-v2','heads':heads,'capabilities':capabilities,'semantics':'normal_reference_distribution_only_not_grade_or_error_probability','grade_prediction_enabled':False,'score_prediction_enabled':False,'production_eligible':False,'training_inputs_sha256':hashlib.sha256((O/'training-inputs.frozen.json').read_bytes()).hexdigest()};path=O/'normal-reference-rebuilt.joblib';assert not path.exists();joblib.dump(artifact,path)
(O/'model-validation.json').write_text(json.dumps({'reports':reports,'grade_capabilities':capabilities,'independent_final_test':False,'Test_inference':False,'validation_sources_used_for_fit':False,'hyperparameters':'fixed SVM nu=.1 gamma=scale; IF 128 trees contamination=.1 seed20261025; no score selection','new_movement_grade_models_fitted':0,'normal_reference_banks_fitted':len(heads),'research_models_fitted':sum(len(h['models']) for h in heads.values()),'deployment_changed':False},ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:{z:v for z,v in r.items() if z!='comparison'}|({'comparison':{a:{b:c for b,c in v.items() if b!='leave_person_out'} for a,v in r['comparison'].items()}} if 'comparison'in r else {}) for k,r in reports.items()},ensure_ascii=False))
