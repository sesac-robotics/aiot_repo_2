"""Pool only the SAME observable frontal stability target across exercises.
Hinged YTW, side-view posture, spine shape and provider error labels are excluded.
"""
from common import B,O,read,save,sha
from cache_sequences import Resolver
from pose_features import relative_trunk
from train_scores import fit,metrics,predictions
from selection import choose
from acceptance import assess
import numpy as np,joblib,collections
import sys
sys.path.insert(0,str(B/'five-exercise-comparison'))
import geometry

def main():
 rows=read(O/'score-inputs.frozen.json');resolver=Resolver();rs=[];excluded=[]
 for r in rows:
  if r['exercise'] not in ['squats','shoulder','ytw']:continue
  n=r['items'][3];g=r['levels'][3]
  desired={'squats':'trunk_balance','shoulder':'frontal_trunk_hold','ytw':'upright_trunk_hold'}[r['exercise']]
  if n['item']!=desired or g is None or not n['eligible'] or n['bank'] not in ['front','front_seated','upright_temporal_v1']:continue
  pairs=resolver.resolve(r)
  if not pairs:excluded.append(r['id']);continue
  chosen=None
  for seq,p in pairs:
   if r['family']=='R2D2' or r['origin']=='error_enrichment':chosen=(seq,p);break
   tilt=geometry.channels(seq)['frontal_tilt_deg'];good=tilt[np.isfinite(tilt)]
   if len(good) and n['feature'] is not None and len(n['feature'])>=7 and np.allclose(np.percentile(good,[5,10,25,50,75,90,95]),n['feature'][:7],rtol=1e-5,atol=1e-4):chosen=(seq,p);break
  if chosen is None:excluded.append(r['id']);continue
  seq,p=chosen;reference=None
  if r['family']=='R2D2':
   d=resolver.team[r['id'][5:]];reference=np.load(B/'output/integrated-five-20261024/team-observations'/d['source_id']/'side.npz')['image_xy'][:3]
  f=relative_trunk(seq,reference)
  if f is None:excluded.append(r['id']);continue
  rs.append(dict(id=r['id'],exercise='shoulder',exercise_origin=r['exercise'],index=3,x=f,target=g,person=r['person'],group=r['group'],family=r['family'],split=r['split'],cache=str(p),annotation_criterion=desired,sources=r['sources']))
 tr=[r for r in rs if r['split']=='train'];va=[r for r in rs if r['split']=='validation'];assert not {r['group'] for r in tr}&{r['group'] for r in va};cv=[];groups=np.array([r['group'] for r in tr]);from sklearn.model_selection import GroupKFold
 for dims in [7,10]:
  data=[dict(r,x=r['x'][:dims]) for r in tr]
  for name in ['tree2','tree4','ridge1','ridge10','ordinal1','grade0.1','grade1','grade10']:
   ps=[];base=[]
   for ti,vi in GroupKFold(4).split(data,groups=groups):
    t=[data[i] for i in ti];v=[data[i] for i in vi];m=fit(t,name);ps+=predictions(m,v);mean=float(np.mean([r['target'] for r in t]));base+=[dict(id=r['id'],group=r['group'],target=r['target'],prediction=mean) for r in v]
   cv.append(dict(name=name,dims=dims,metrics=metrics(ps),constant_baseline=metrics(base),predictions=ps))
 best=choose(cv);t=[dict(r,x=r['x'][:best['dims']]) for r in tr];v=[dict(r,x=r['x'][:best['dims']]) for r in va];m=fit(t,best['name']);vp=predictions(m,v);gate=assess(best['predictions'],vp,best['constant_baseline']);joblib.dump(dict(model=m,dims=best['dims'],feature_schema='source_ready_referenced_frontal_tilt_change_v1',criterion='observed_frontal_trunk_stability_only',supported_grades=sorted({r['target'] for r in tr}),training_ids=[r['id'] for r in tr],training_people=sorted({r['person'] for r in tr}),production_replacement=False),O/'shared-frontal-trunk-candidate.joblib');save('shared-trunk-inputs.frozen.json',rs);save('shared-trunk-training.json',dict(train_n=len(tr),validation_n=len(va),grade_counts=dict(collections.Counter(r['target'] for r in tr)),source_exercise_counts=dict(collections.Counter(r['exercise_origin'] for r in tr)),selected={k:v for k,v in best.items() if k!='predictions'},group_cv=best['metrics'],fixed_development_validation=metrics(vp),acceptance=gate,validation_predictions=vp,candidate_cv=[{k:v for k,v in c.items() if k!='predictions'} for c in cv],excluded=excluded,labels_are_existing_direct_item_grades_not_provider_errors=True,final_test_used=False,production_replacement=False));print('SHARED_TRUNK',len(tr),metrics(vp),gate,flush=True)
if __name__=='__main__':main()
