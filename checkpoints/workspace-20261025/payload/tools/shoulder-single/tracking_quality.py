"""Shoulder-only research tracking-quality head; never returns a posture grade."""
import numpy as np
VERSION='shoulder-four-joint-tracking-shape-v1'
FEATURE_NAMES=['shoulder_width_over_axis','hip_width_over_axis','shoulder_hip_width_ratio','left_side_over_axis','right_side_over_axis','side_length_ratio','diagonal_1_over_axis','diagonal_2_over_axis','shoulder_axis_alignment','hip_axis_alignment']
def features(xy):
 x=np.asarray(xy,float)
 if x.shape!=(33,2) or not np.isfinite(x[[11,12,23,24]]).all():return None
 a,b,c,d=x[[11,12,23,24]];axis=(a+b-c-d)/2;scale=np.linalg.norm(axis)
 if scale<1:return None
 sw=np.linalg.norm(a-b);hw=np.linalg.norm(c-d);left=np.linalg.norm(a-c);right=np.linalg.norm(b-d)
 if min(sw,hw,left,right)<1e-6:return None
 return np.array([sw/scale,hw/scale,sw/hw,left/scale,right/scale,left/right,np.linalg.norm(a-d)/scale,np.linalg.norm(b-c)/scale,abs(np.dot(a-b,axis))/(sw*scale),abs(np.dot(c-d,axis))/(hw*scale)])
def assess(model,xy,view='front'):
 if view!='front':return dict(status='held',reason='unvalidated_view',posture_grade=None)
 f=features(xy)
 if f is None:return dict(status='held',reason='required_four_joints_unobserved',posture_grade=None)
 p=float(model.predict_proba([f])[0,list(model.classes_).index(1)])
 return dict(status='tracking_review_required' if p>=.5 else 'tracking_screen_passed',tracking_error_probability=p,posture_grade=None,posture_score=None,anatomical_correctness_guaranteed=False,production_eligible=False)
