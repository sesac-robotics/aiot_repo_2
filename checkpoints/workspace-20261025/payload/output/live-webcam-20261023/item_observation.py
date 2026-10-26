"""Per-item observation requirements; head, ankles and feet never gate these items."""
import numpy as np
VERSION='front-item-observation-v1'
REQUIREMENTS={
 'trunk_balance':{'joints':[11,12,23,24],'minimum_frames':10,'coverage':.8},
 'standing_return':{'joints':[11,12,23,24],'minimum_frames':10,'coverage':.8},
 'knee_inward':{'joints':[23,24,25,26],'minimum_frames':10,'coverage':.8},
 'descent_range':{'joints':[11,12,23,24,25,26],'minimum_frames':10,'coverage':.8},
}

def quant(v):return [float(t) for t in np.percentile(np.asarray(v,float),[10,50,90])]

def observed_features(xy,width=None,height=None):
 x=np.asarray(xy,float).copy();out={}
 if x.ndim!=3 or x.shape[1:]!=(33,2) or not len(x):return out
 if width is not None and height is not None:
  outside=(x[:,:,0]<0)|(x[:,:,0]>=width)|(x[:,:,1]<0)|(x[:,:,1]>=height);x[outside]=np.nan
 def eligible(key):
  spec=REQUIREMENTS[key];mask=np.isfinite(x[:,spec['joints']]).all((1,2))
  # A missing starting reference must not be replaced by the squat bottom.
  reference_window=max(3,min(10,int(np.ceil(len(x)*.2))))
  good=mask.sum()>=spec['minimum_frames'] and mask.mean()>=spec['coverage'] and mask[:reference_window].sum()>=3
  return mask,good
 torso,ok=eligible('trunk_balance')
 if ok:
  t=x[torso];s=t[:,[11,12]].mean(1);h=t[:,[23,24]].mean(1);scale=np.median(np.linalg.norm(h-s,axis=1))
  if scale>=1:
   tilt=np.degrees(np.arctan2(h[:,0]-s[:,0],h[:,1]-s[:,1]));shoulder=np.degrees(np.arctan2(t[:,12,1]-t[:,11,1],abs(t[:,12,0]-t[:,11,0])))
   vals=quant(abs(tilt-np.median(tilt[:3])))+quant(abs(shoulder-np.median(shoulder[:3])))+quant(abs(tilt))
   out['trunk_balance']=dict(x=vals,scalar=vals[2],higher=False,coverage=float(torso.mean()))
   if torso[-3:].sum()>=2:
    hy=h[:,1];n=min(10,max(3,len(hy)//4));delta=abs(np.median(np.sort(hy[:n])[:3])-np.median(np.sort(hy[-n:])[:3]))/scale
    out['standing_return']=dict(x=[float(delta)],scalar=float(delta),higher=False,coverage=float(torso.mean()))
 knees,ok=eligible('knee_inward')
 if ok:
  a=x[knees];hipwidth=abs(a[:,24,0]-a[:,23,0]);ratio=abs(a[:,26,0]-a[:,25,0])/np.maximum(hipwidth,1);base=np.median(ratio[:3]);relative=ratio/max(.01,base);q=quant(relative)
  out['knee_inward']=dict(x=q+quant(ratio),scalar=q[0],higher=True,coverage=float(knees.mean()))
 # Depth remains under the existing synchronized-side-label estimator; no
 # depth or heel labels are invented from a body-only sample.
 return out
