from pathlib import Path
import os,json,sys,datetime,hashlib,collections
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import joblib,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
B=Path('/home/jayden/SeSACPAI-Mid');sys.path.insert(0,str(B/'tools/error-enrichment'));from split_guard import assert_training_allowed
P=B/'output/shoulder-work-20261024/normal-reference-v2';O=B/'output/shoulder-work-20261024/normal-reference-v3';O.mkdir(exist_ok=True);assert not (O/'plan.frozen.json').exists()
dec=json.loads((P/'visual-chain-decisions.json').read_text());old=json.loads((P/'inputs.frozen.json').read_text());rows=[r for r in old if r['id'] not in dec['quarantine']];assert len(old)-len(rows)==7
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(O/'plan.frozen.json').write_text(json.dumps(dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_inputs_sha256=sha(P/'inputs.frozen.json'),chain_decisions_sha256=sha(P/'visual-chain-decisions.json'),quarantined_ids=list(dec['quarantine']),configuration_unchanged=True,no_source_label_changes=True,feature_qa_before_this_fit=True),ensure_ascii=False,indent=2)+'\n')
(O/'inputs.frozen.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
heads={};metrics={};make=lambda:make_pipeline(StandardScaler(),OneClassSVM(nu=.05,gamma='scale'))
for key in sorted({(r['item'],r['bank']) for r in rows}):
 tr=[r for r in rows if (r['item'],r['bank'])==key and r['split']=='train'];va=[r for r in rows if (r['item'],r['bank'])==key and r['split']=='validation'];assert_training_allowed(tr);people=sorted({r['person'] for r in tr});k='/'.join(key)
 if len(people)<2:metrics[k]=dict(status='held',reason='fewer_than_two_Train_people',train_rows=len(tr),train_people=people);continue
 m=make().fit([r['feature'] for r in tr]);heads[key]=m;loo=[]
 for person in people:
  a=[r for r in tr if r['person']!=person];b=[r for r in tr if r['person']==person];fit=make().fit([r['feature'] for r in a]);pred=fit.predict([r['feature'] for r in b]);loo.extend([dict(id=r['id'],person=person,normal_reference_inlier=bool(p==1)) for r,p in zip(b,pred)])
 vp=m.predict([r['feature'] for r in va]).tolist() if va else []
 metrics[k]=dict(status='normal_reference_fitted',train_rows=len(tr),train_people=people,development_normal_rows=len(va),development_normal_inlier_count=sum(v==1 for v in vp),leave_person_out_normal_rows=len(loo),leave_person_out_normal_inlier_count=sum(r['normal_reference_inlier'] for r in loo),leave_person_out=loo,error_label_count=0,acceptance='research_only_generalization_not_established')
bundle=dict(schema='shoulder-normal-reference-v3',heads=heads,semantics='normal_training_distribution_support_only',grade_prediction_enabled=False,score_prediction_enabled=False,error_probability_enabled=False,production_eligible=False,feature_schema_by_head={str(k):('frontal-four-joint-absolute-and-window-initial-relative-tilt-v1' if k[0].endswith('window_reference') else 'shoulder-ready-return-relative-v3') for k in heads})
joblib.dump(bundle,O/'normal-reference.joblib');(O/'metrics.json').write_text(json.dumps(dict(metrics=metrics,fixed_configuration={'nu':.05,'gamma':'scale'},source_roles_preserved=True,expert_adjudicated=False,test_inference=False,production_eligible=False),ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:{a:b for a,b in v.items() if a not in ['leave_person_out','train_people']} for k,v in metrics.items()},ensure_ascii=False))
