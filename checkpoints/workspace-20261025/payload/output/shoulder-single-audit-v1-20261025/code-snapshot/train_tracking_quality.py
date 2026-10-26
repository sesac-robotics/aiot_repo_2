"""Train on original-video reviewed Train frames, grouped by person; no grade fitting."""
from pathlib import Path
import os,sys,json,hashlib,collections
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesClassifier
from tracking_quality import features,VERSION,FEATURE_NAMES
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-single-audit-v1-20261025';sys.path.insert(0,str(B/'tools/error-enrichment'));from split_guard import assert_training_allowed
sys.path.insert(0,str(B/'tools/shoulder-work'));from observation_features_v4 import quality
rows={r['id']:r for r in json.loads((O/'train-provenance-audit.json').read_text())['rows']};prep=json.loads((O/'visual-preparation.json').read_text())
# Direct visual judgments from the original-source/cache overlays; 1 means incorrect tracking,
# never severity 1. Uncertain/back-facing panels are explicitly excluded, not normal-filled.
neg={'subject_001_shoulder_bad_front':[88],'subject_001_shoulder_good_front':[177,178], 'subject_006_shoulder_good_front':[135,177,178,180], 'subject_010_shoulder_good_front':[28], 'subject_014_shoulder_good_front':[20], 'subject_019_shoulder_bad_front':[12,39,50,51], 'subject_019_shoulder_good_front':[30,38,39]}
uncertain={'subject_006_shoulder_bad_front':[0,90,177,178,180], 'subject_007_shoulder_bad_front':[39,45,50], 'subject_026_shoulder_bad_front':[10], 'subject_013_shoulder_bad_front':[12]}
labels=[];excluded=[]
for s in prep:
 r=rows[s['id']]
 # Actual rear-view examples are not silently assigned to the front-bank quality model.
 if r['person'] in ['shoulder_005','shoulder_008','shoulder_009']:
  excluded.append({'id':r['id'],'reason':'rear_view_requires_separate_bank'});continue
 assert r['split']==r['canonical_split']=='train'
 assert hashlib.sha256(Path(r['source_path']).read_bytes()).hexdigest()==r['source_sha256']
 z=np.load(r['cache'])
 for i,frame in zip(s['sample_indices'],s['source_frames']):
  if frame in uncertain.get(r['id'],[]):excluded.append({'id':r['id'],'native_frame':frame,'reason':'visual_tracking_uncertain'});continue
  x=z['image_xy'][i];f=features(x)
  if f is None:excluded.append({'id':r['id'],'native_frame':frame,'reason':'required_joint_missing'});continue
  q=quality(np.array([0]),x[None])
  labels.append(dict(id=r['id']+'_tracking_frame_'+str(frame),source_id=r['id'],source_sha256=r['source_sha256'],sources=[r['source_sha256']],person=r['person'],split='train',native_frame=frame,cache_sample_index=i,cache=r['cache'],cache_sha256=r['cache_sha256'],feature=f.tolist(),tracking_error=int(frame in neg.get(r['id'],[])),label_target='four_joint_tracking_misconnection_not_movement_severity',expert_adjudicated=False,reviewer='assistant_direct_original_video_overlay_review',evidence=str(O/s['page']),baseline_quality_flag=bool(q['flags'])))
assert_training_allowed(labels)
# Both source aliases and all frames from a participant stay in the same fold.
assert len({(r['source_sha256'],r['native_frame']) for r in labels})==len(labels)
def make():return ExtraTreesClassifier(n_estimators=128,min_samples_leaf=2,class_weight='balanced',random_state=20261025,n_jobs=1,max_features=None)
def metrics(records,key):
 return {'tracking_errors':sum(r['tracking_error']==1 for r in records),'tracking_errors_flagged':sum(r['tracking_error']==1 and r[key] for r in records),'visually_accepted_frames':sum(r['tracking_error']==0 for r in records),'visually_accepted_frames_flagged':sum(r['tracking_error']==0 and r[key] for r in records)}
cv=[]
for person in sorted({r['person'] for r in labels}):
 tr=[r for r in labels if r['person']!=person];te=[r for r in labels if r['person']==person];m=make().fit([r['feature'] for r in tr],[r['tracking_error'] for r in tr]);p=m.predict_proba([r['feature'] for r in te])[:,list(m.classes_).index(1)]
 cv.extend([dict(id=r['id'],person=person,tracking_error=r['tracking_error'],baseline=r['baseline_quality_flag'],candidate=bool(v>=.5),probability=float(v)) for r,v in zip(te,p)])
m=make().fit([r['feature'] for r in labels],[r['tracking_error'] for r in labels]);joblib.dump({'schema':VERSION,'feature_names':FEATURE_NAMES,'model':m,'target':'tracking_error_not_posture_grade','grade_prediction_enabled':False,'score_prediction_enabled':False,'production_eligible':False},O/'tracking-quality-candidate.joblib')
result={'rows':len(labels),'people':len({r['person'] for r in labels}),'errors_people':sorted({r['person'] for r in labels if r['tracking_error']}),'baseline':metrics(cv,'baseline'),'candidate':metrics(cv,'candidate'),'leave_person_out':cv,'selection':'fixed_128_trees_leaf2_threshold0.5_no_hyperparameter_search','scope':'exposed_Train_development_not_independent_final_Test','frame_labels_not_movement_grades':True,'grade_model_fit':False,'production_changed':False}
for name,obj in [('tracking-labels.frozen.json',labels),('tracking-review-exclusions.json',excluded),('tracking-model-metrics.json',result)]:
 p=O/name
 if p.exists():raise ValueError('Immutable experiment already exists: '+name)
 p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='leave_person_out'},ensure_ascii=False))
