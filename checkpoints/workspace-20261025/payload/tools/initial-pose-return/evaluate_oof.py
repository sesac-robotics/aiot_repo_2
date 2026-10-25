"""Run the service path on whole Train-source streams with the assessed person left out."""
import json,sys,collections,copy,datetime,os,cv2
from pathlib import Path
from fractions import Fraction
import numpy as np,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
B=Path('/home/jayden/SeSACPAI-Mid');O=Path(os.environ.get('INITIAL_POSE_RUN',str(B/'output/initial-pose-return-v4-20261024')));sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'team-posture-scoring/Server/AIService')]
from common import sha
from split_guard import assert_training_allowed,POLICY
from app.scoring.engine import ScoringEngine
from app.scoring.sequence import PoseSequence
from app.scoring.initial_pose_return import pose
rows=json.loads((O/'state-inputs.frozen.json').read_text());training=json.loads((O/'state-training.json').read_text());models=joblib.load(O/'initial-pose-return-bundle.joblib');reports={tuple(h['key']):h for h in training['heads'] if h['status']=='trained_experimental'};inventory={r['id']:r for r in json.loads((B/'output/r2d2-review-20261024/inventory.json').read_text())};sources=json.loads((O/'source-intake.frozen.json').read_text());engine=ScoringEngine();results=[];folds={};feature_checks=0
for r in rows:
 z=np.load(r['cache']);ix=np.flatnonzero(z['frame_numbers']==r['observed_frame']);assert len(ix)==1;assert np.allclose(pose(r['exercise'],z['image_xy'][ix[0]]),r['x']);feature_checks+=1
for src in sources:
 person=src['person'] if src.get('person_is_canonical') else 'r2d2_'+src['person'];heads={};ids={}
 for key,h in models['heads'].items():
  if key[0]!=src['exercise']:continue
  cache_key=(key,person)
  if cache_key not in folds:
   field,target=key[1:];rs=[r for r in rows if r['exercise']==key[0] and r['person']!=person and r[field] is not None and r[target] is not None];assert_training_allowed(rs);assert all(r['person']!=person for r in rs);name=reports[key]['selected'];m=make_pipeline(StandardScaler(),LogisticRegression(C=.3 if name=='log0.3' else 3,max_iter=1500,random_state=241024)) if name.startswith('log') else make_pipeline(StandardScaler(),SVC(C=float(name[3:]),probability=True,random_state=241024)) if name.startswith('svm') else ExtraTreesClassifier(n_estimators=128,min_samples_leaf=2,max_features=1.,random_state=241024,n_jobs=1);counts=collections.Counter((r[target],r['person']) for r in rs);ys=collections.Counter(y for y,p in counts);w=np.array([1/(ys[r[target]]*counts[r[target],r['person']]) for r in rs]);w*=len(w)/w.sum() if name.startswith('svm') else 1;kw={'logisticregression__sample_weight':w} if name.startswith('log') else {'svc__sample_weight':w} if name.startswith('svm') else {'sample_weight':w};m.fit([r[field] for r in rs],[r[target] for r in rs],**kw);folds[cache_key]=(dict(h,model=m),[r['id'] for r in rs])
  heads[key],ids[str(key)]=folds[cache_key]
 p=Path(src['cache']) if src.get('cache') else B/'output/integrated-five-20261024/team-observations'/src['id']/'side.npz';z=np.load(p);s=inventory.get(src['id']);fps=float(Fraction(src['fps']));times=z['frame_numbers']/fps
 if s:
  end=min(src['duration']-.05,11.8);mask=(times<=end);rot=abs(float(s.get('rotation',[{}])[0].get('rotation',0)))%180;width=s['height'] if rot==90 else s['width'];height=s['width'] if rot==90 else s['height']
 else:
  mask=(times>=src['review_start'])&(times<=src['review_end']);cap=cv2.VideoCapture(src['source']);width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release();assert width>0 and height>0
 xy=z['image_xy'][mask];frames=z['frame_numbers'][mask];meta=json.loads(p.with_suffix('.json').read_text());assert meta['config']['source_sha256']==src['source_sha256'];seq=PoseSequence(xy,frames,fps,width,height);protocol={'pushups':'arm_return_v1','shoulder':'seated_rack_v1','ytw':'upright_Y_T_W_v1','plank':'support_hold_v1','squats':'standing_return_v1'}[src['exercise']]
 result=engine.analyze_initial_return(src['exercise'],seq,state_models=dict(models,heads=heads),view='side' if src['exercise'] in ['pushups','plank'] else 'front',protocol=protocol)
 results.append(dict(source_id=src['id'],exercise=src['exercise'],person=person,start=float(times[mask][0]),end=float(times[mask][-1]),held_person_not_trained=True,fold_training_ids=ids,result=result));print('STREAM_OOF',src['id'],result.get('state'),result.get('phase_order'),flush=True)
(O/'stream-oof-v2-evaluation.json').write_text(json.dumps(dict(schema='initial_state_temporal_development_v1',completed=True,Test_used=False,source_policy_sha256=sha(POLICY),feature_value_parity_checked_frames=feature_checks,group_cv_selection_not_nested_independent_validation=True,models_selected_before_stream_evaluation=True,streams=results,not_a_return_grade_accuracy_claim=True),indent=2));print('STREAM_COMPLETE',len(results),flush=True)
