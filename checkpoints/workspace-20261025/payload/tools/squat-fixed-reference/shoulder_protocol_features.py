"""Observed side-arm protocol geometry, independent of severity or provider tags."""
import numpy as np
VERSION='observed-single-arm-protocol-quantiles-v1'
NAMES=[f'{name}_{q}' for name in ['elevation','elbow','wrist_up'] for q in ['q10','q50','q90','span90_10']]
def angle(a,b,c):
 u=a-b;v=c-b;den=np.linalg.norm(u,axis=-1)*np.linalg.norm(v,axis=-1)
 return np.degrees(np.arccos(np.clip(np.sum(u*v,axis=-1)/np.maximum(den,1e-8),-1,1)))
def extract(xy,width,height):
 x=np.asarray(xy,float).copy()
 if x.ndim!=3 or x.shape[1:]!=(33,2) or len(x)<10:return dict(eligible=False,reason='insufficient_actual_samples')
 bad=(x[:,:,0]<.02*width)|(x[:,:,0]>.98*width)|(x[:,:,1]<.02*height)|(x[:,:,1]>.98*height)
 x[bad]=np.nan;candidates=[]
 for side,js in [('left',[11,23,13,15]),('right',[12,24,14,16])]:
  a=x[:,js];valid=np.isfinite(a).all((1,2));scale=np.linalg.norm(a[:,0]-a[:,1],axis=1);valid&=scale>5
  candidates.append((int(valid.sum()),side,js,valid))
 count,side,js,valid=max(candidates,key=lambda c:c[0])
 if count<10 or valid.mean()<.8:return dict(eligible=False,reason='single_arm_chain_not_observed_80percent',observed_count=count,sample_count=len(x))
 a=x[valid][:,js];scale=np.linalg.norm(a[:,0]-a[:,1],axis=1);up=(a[:,0]-a[:,1])/scale[:,None]
 curves=[angle(a[:,1],a[:,0],a[:,2])/180,angle(a[:,0],a[:,2],a[:,3])/180,
         np.sum((a[:,3]-a[:,0])*up,axis=1)/scale]
 f=[]
 for c in curves:
  q=np.percentile(c,[10,50,90]);f.extend([*q,q[2]-q[0]])
 return dict(eligible=True,x=np.asarray(f).tolist(),side=side,required_joints=js,coverage=float(valid.mean()),
             valid_indices=np.flatnonzero(valid).tolist(),feature_version=VERSION)
