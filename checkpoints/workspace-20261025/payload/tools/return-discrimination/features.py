"""Initial-pose deficits relative to observed support; no severity/source features."""
import numpy as np
VERSION='standing-support-extension-v1'
NAMES=['hip_height_deficit_mean','hip_height_deficit_max','shoulder_height_deficit_mean','shoulder_height_deficit_max','knee_extension_deficit_mean','knee_extension_deficit_max','hip_extension_deficit_mean','hip_extension_deficit_max','torso_vertical_deficit_mean','torso_vertical_deficit_max']
def angle(a,b,c):
 u=a-b;v=c-b;n=np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)
 if np.any(n<1):return None
 return np.degrees(np.arccos(np.clip(np.sum(u*v,axis=-1)/n,-1,1)))
def values(x):
 s,h,k,a=[x[:,i] for i in range(4)];leg=np.linalg.norm(h-k,axis=1)+np.linalg.norm(k-a,axis=1);torso=np.linalg.norm(s-h,axis=1)
 if np.any(leg<2) or np.any(torso<2):return None
 knee=angle(h,k,a);hip=angle(s,h,k)
 if knee is None or hip is None:return None
 return np.column_stack([(h[:,1]-a[:,1])/leg,(s[:,1]-a[:,1])/leg,-knee/180,-hip/180,(s[:,1]-h[:,1])/torso])
def extract(xy,reference_xy,base):
 if base is None:return None
 x=np.asarray(xy,float);r=np.asarray(reference_xy,float);t=base['return_index'];out=[];sides=[]
 for name,chain in [('left',[11,23,25,27]),('right',[12,24,26,28])]:
  a=r[:,chain];b=x[t:t+3,chain]
  if len(b)!=3 or not np.isfinite(b).all() or np.isfinite(a).all((1,2)).mean()<.8:continue
  a=a[np.isfinite(a).all((1,2))]
  if len(a)<3:continue
  av=values(a);bv=values(b)
  if av is None or bv is None:continue
  leg=lambda q:np.median(np.linalg.norm(q[:,1]-q[:,2],axis=1)+np.linalg.norm(q[:,2]-q[:,3],axis=1))
  ratio=leg(b)/leg(a)
  if not .7<=ratio<=1.3:continue
  out.append(np.median(bv,axis=0)-np.median(av,axis=0));sides.append(name)
 if not out:return None
 q=np.asarray(out);return dict(x=np.column_stack([q.mean(axis=0),q.max(axis=0)]).flatten().tolist(),sides=sides,return_index=t,bottom_index=base['bottom_index'],feature_version=VERSION)
