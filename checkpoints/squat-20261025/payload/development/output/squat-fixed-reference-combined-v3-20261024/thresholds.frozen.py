"""Explicit ordered scalar thresholds; fitted only from protected Train labels.
No estimator output or native-condition AI probability is read by this module.
"""
import sys,collections
from pathlib import Path
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');sys.path.insert(0,str(B/'tools/error-enrichment'))
from split_guard import assert_training_allowed
from train_scores import weights

def transform_input(bank,criterion,features):
 a=np.asarray(features,float);a=a[None,:] if a.ndim==1 else a
 if '_aihub_transfer' in bank:a=a[:,:-1]
 if criterion=='support_continuity' and a.shape[1]==7:return (a[:,6]-a[:,0])[:,None]
 if 'support_clearance' in bank and criterion=='leg_and_body_support':
  measures=a[:,[1,8,15]]/np.array([165.,.12,.03]);return np.minimum.reduce(measures,axis=1)[:,None]
 return a

def allowed_indices(bank,criterion,dims):
 # Learned native-condition probabilities are last; forbid them for Rule Base.
 native='_aihub_transfer' in bank
 d=dims-1 if native else dims
 if criterion=='support_continuity' and d==7:return [0]
 if 'support_clearance' in bank and criterion=='leg_and_body_support':return [0]
 if criterion=='elbow_sagittal_position' and d==7:return [5]
 if 'temporal' in bank:return []
 if 'compact_shape' in bank:return [15]
 if 'arm_pose' in bank and d==98:
  return [{'y_reach':95,'t_reach':96,'w_shape':97}.get(criterion,95)]
 if 'compact_press' in bank:return [15]
 if 'compact_timing' in bank:return [1,2]
 if 'body_alignment' in bank:return [4,9]
 if bank=='height_v4':return [0,1,2]
 if bank in ['height_upper_reversal_v1','height_reference_pose_v1']:return [0,1,2,3]
 if bank=='standing_initial_context_v1':return [0,1,2,3]
 if bank in ['standing_fixed_initial_height_v2','standing_fixed_initial_context_v3']:return [0,1,2,3]
 if bank=='standing_support_extension_v1':return [0,1,2,3]  # Signed hip/shoulder deficits; angles alone cannot grade height return.
 if 'gap_relative' in bank:return [0,1,7,8]
 if 'trunk_relative' in bank:return [5,6]
 if 'semantic_roi' in bank:
  return {'trunk_line':[1,2],'descent_range':[0,1],'return_extension':[0,1],'support':[0,1,3]}[criterion]
 if 'chest_displacement' in bank:return [0,3]
 if 'support_clearance' in bank:return list(range(min(6,d)))
 if d in [7,8]:
  idx=1 if criterion in ['trunk_line','descent_range','support','leg_and_body_support','elbow_sagittal_position'] else 5
  return [idx,3]
 return sorted(set([0,min(1,d-1),min(3,d-1),min(5,d-1)]))

def predict(rule,x):
 a=np.asarray(x,float)
 if rule.get('constant') is not None:return np.full(len(a),rule['constant'],dtype=float)
 values=a[:,rule['index']]*rule['direction'];cuts=np.asarray(rule['thresholds']);grades=np.asarray(rule['grades']);return grades[np.searchsorted(cuts,values,side='right')].astype(float)

def fit(rows,index,direction):
 assert_training_allowed(rows)
 x=np.asarray([r['x'][index] for r in rows])*direction;y=np.asarray([r['target'] for r in rows]);grades=np.unique(y)
 if len(grades)==1:return dict(index=index,direction=direction,constant=float(grades[0]),thresholds=[],grades=grades.tolist())
 w=weights(rows);normal=y==2
 for mask in [normal,~normal]:
  if mask.any():w[mask]/=w[mask].sum()
 w/=w.sum()
 unique=np.unique(x);cuts=(unique[:-1]+unique[1:])/2
 if len(cuts)>30:cuts=np.quantile(cuts,np.linspace(0,1,30))
 gap=max(1.,float(np.ptp(x)));cuts=np.r_[unique[0]-gap,cuts,unique[-1]+gap]
 best=None
 choices=[(float(c),) for c in cuts] if len(grades)==2 else [(float(a),float(b)) for a in cuts for b in cuts if a<=b]
 for candidate in choices:
  pred=grades[np.searchsorted(candidate,x,side='right')]
  binary=float(np.sum(w*((pred<2)!=(y<2))));mae=float(np.sum(w*abs(pred-y)))
  rank=(binary,mae,sum(abs(t) for t in candidate))
  if best is None or rank<best[0]:best=(rank,candidate)
 return dict(index=index,direction=direction,thresholds=list(best[1]),grades=grades.tolist(),constant=None)
