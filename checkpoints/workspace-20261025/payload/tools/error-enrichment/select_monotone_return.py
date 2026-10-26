"""Supervised monotone standing recognizer, selected on TRAINING return CV.
Normal confidence cannot decrease solely because the hip/shoulder becomes higher,
torso becomes more extended, or knees become straighter, within observation gates.
No hand-written score thresholds or rule grades serve as targets.
"""
from standing_state import *
from sklearn.ensemble import HistGradientBoostingClassifier
from train_scores import metrics

def mono_fit(rows,dims,leaf,reg):
 assert_training_allowed(rows)
 signs=np.array([-1,-1,1,-1,-1]);m=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=leaf,min_samples_leaf=3,l2_regularization=reg,monotonic_cst=signs[dims].tolist(),random_state=241024);m.fit([[r['x'][i] for i in dims] for r in rows],[r['target'] for r in rows],sample_weight=weights(rows));return m

def attempts(refs):
 labels={r['id']:r for r in read(B/'output/squat-return-20261023/labels.frozen.json')['labels']};prior=read(B/'output/squat-return-20261023/feature-rows.json');cache={};rs=[];excluded=[]
 for r in prior:
  d=labels[r['id']];source=Path(d['path']).stem;ref=refs.get(source)
  if ref is None or d['first_frame']<=ref['calibration_frame']:excluded.append(dict(id=r['id'],reason='no_preceding_observed_calibration'));continue
  if source not in cache:cache[source]=load_frames(dict(review_id=source))
  xy,fs=cache[source];seq=xy[(fs>=d['first_frame'])&(fs<=d['last_frame'])];xs=[frame_x(x,ref) for x in seq];valid=np.array([x is not None for x in xs]);drop=np.array([x[0] if x else np.nan for x in xs])
  if len(xs)<10 or valid.mean()<.8:excluded.append(dict(id=r['id'],reason='required_observation_insufficient'));continue
  bottom=int(np.nanargmax(drop));tail=max(bottom+1,len(xs)-max(3,len(xs)//4));windows=[i for i in range(tail,len(xs)-2) if valid[i:i+3].all()]
  if not windows:excluded.append(dict(id=r['id'],reason='post_descent_return_unobserved'));continue
  rs.append(dict(id=r['id'],person=r['person'],group=r['person'],split=r['split'],target=r['target'],xs=xs,valid=valid,windows=windows,calibration_frame=ref['calibration_frame']))
 return rs,excluded

def predict_attempt(m,r,dims):
 ps=np.full(len(r['xs']),np.nan);good=np.flatnonzero(r['valid']);ps[good]=m.predict_proba([[r['xs'][j][i] for i in dims] for j in good])[:,list(m.classes_).index(1)];p=max(float(np.median(ps[i:i+3])) for i in r['windows']);return dict(id=r['id'],person=r['person'],group=r['group'],split=r['split'],target=r['target'],prediction=1+p,standing_probability=p)

def main():
 frames,refs=prepare();tr=[r for r in frames if r['split']=='train'];va=[r for r in frames if r['split']=='validation'];rs,excluded=attempts(refs);train=[r for r in rs if r['split']=='train'];validation=[r for r in rs if r['split']=='validation'];reports=[]
 for dims in [[0],[0,1,2],[0,3,4],[0,1,2,3,4]]:
  for leaf,reg in [(3,.5),(5,.5),(3,2),(5,2)]:
   ps=[]
   for person in sorted({r['person'] for r in train}):
    m=mono_fit([r for r in tr if r['person']!=person],dims,leaf,reg);ps +=[predict_attempt(m,r,dims) for r in train if r['person']==person]
   reports.append(dict(name='monotone',dims=dims,leaf=leaf,regularization=reg,metrics=metrics(ps),predictions=ps));print('RETURN_MONOTONE_CV',dims,leaf,reg,metrics(ps),flush=True)
 eligible=[r for r in reports if r['metrics']['error_recall']>=.65 and r['metrics']['normal_false_alarm']<=.15];best=min(eligible,key=lambda r:r['metrics']['balanced_group_MAE']) if eligible else min(reports,key=lambda r:((1-r['metrics']['error_recall']+r['metrics']['normal_false_alarm'])/2,r['metrics']['balanced_group_MAE']));m=mono_fit(tr,best['dims'],best['leaf'],best['regularization']);vp=[predict_attempt(m,r,best['dims']) for r in validation];framep=m.predict_proba([[r['x'][i] for i in best['dims']] for r in va])[:,list(m.classes_).index(1)];bundle=dict(model=m,dims=best['dims'],name='supervised_monotone_standing_state',feature_schema='relative_hip_y_shoulder_y_torso_ratio_relative_knee_flex_relative_leg_height_v2',feature_names=['relative_hip_y','relative_shoulder_y','torso_length_ratio','relative_knee_flexion','relative_leg_height_shortfall'],training_ids=[r['id'] for r in tr],training_people=sorted({r['person'] for r in tr}),selected_on='training_person_excluded_return_attempts_only',monotonic_constraints=np.array([-1,-1,1,-1,-1])[best['dims']].tolist(),production_replacement=False,frame_state_not_repetition_failure=True,supported_return_grades=[1,2]);joblib.dump(bundle,O/'standing-monotone-candidate.joblib');save('standing-monotone-training.json',dict(selected={k:v for k,v in best.items() if k!='predictions'},candidates=[{k:v for k,v in r.items() if k!='predictions'} for r in reports],train_frames=len(tr),validation_frames=len(va),frame_development_validation=metric(va,framep),group_cv=best['metrics'],fixed_development_validation=metrics(vp),training_predictions=best['predictions'],validation_predictions=vp,excluded=excluded,selected_without_using_development_validation=True,final_test_used=False,severe_grade_zero_supported=False,production_replacement=False,limitations=['Existing developer validation repeatedly reviewed; not an independent final test.','Only one person has fixed validation return errors.','Labels are visual drafts.']));print('MONOTONE_FINAL',best['dims'],best['metrics'],metrics(vp),flush=True)
if __name__=='__main__':main()
