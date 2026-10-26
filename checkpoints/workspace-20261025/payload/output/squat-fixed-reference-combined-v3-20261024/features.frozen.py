"""Keep initial leg scale fixed so a shortened return cannot normalize itself away."""
import numpy as np
VERSION='standing-fixed-initial-support-height-v2'
NAMES=['hip_height_deficit_mean','hip_height_deficit_max','shoulder_height_deficit_mean','shoulder_height_deficit_max']
def extract(xy,reference_xy,base):
 if base is None:return None
 x=np.asarray(xy,float);r=np.asarray(reference_xy,float);t=base['return_index'];values=[];sides=[];ratios=[]
 for side,js in [('left',[11,23,25,27]),('right',[12,24,26,28])]:
  a=r[:,js];b=x[t:t+3,js]
  if len(b)!=3 or not np.isfinite(b).all():continue
  good=np.isfinite(a).all((1,2));a=a[good]
  if len(a)<3 or good.mean()<.8:continue
  p=np.median(a,axis=0);q=np.median(b,axis=0)
  leg=lambda z:float(np.linalg.norm(z[1]-z[2])+np.linalg.norm(z[2]-z[3]))
  L=leg(p);ratio=leg(q)/L if L>2 else 0
  if not .7<=ratio<=1.3:continue
  # The support translation is removed from each pose, but both heights use
  # the SAME pre-exercise scale. Return leg shortening is not camera zoom.
  vals=[((q[j,1]-q[3,1])-(p[j,1]-p[3,1]))/L for j in [1,0]]
  values.append(vals);sides.append(side);ratios.append(ratio)
 if not values:return None
 a=np.asarray(values)
 return dict(x=np.column_stack([a.mean(0),a.max(0)]).flatten().tolist(),return_index=t,bottom_index=base['bottom_index'],sides=sides,observed_return_leg_ratios=ratios,feature_version=VERSION)
