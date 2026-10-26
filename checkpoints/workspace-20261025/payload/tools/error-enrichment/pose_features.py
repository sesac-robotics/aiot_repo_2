"""Unscored geometric observations; never provider hints or model/rule grades."""
import sys,numpy as np
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');sys.path.insert(0,str(B/'five-exercise-comparison'));import geometry

def arm_pose(seq):
 required=[11,12,13,14,15,16,23,24];valid=np.isfinite(seq[:,required]).all((1,2))
 if len(seq)<10 or valid.mean()<.8:return None
 # Discard unobserved samples, retaining their original timestamps in dwell counts.
 ch=geometry.channels(seq);le=ch['left_elevation_deg']/180;re=ch['right_elevation_deg']/180;lb=ch['left_elbow_deg']/180;rb=ch['right_elbow_deg']/180
 hip=seq[:,[23,24]].mean(1);sh=seq[:,[11,12]].mean(1);axis=sh-hip;scale=np.linalg.norm(axis,axis=1);up=axis/np.maximum(scale[:,None],1);side=np.stack([-up[:,1],up[:,0]],axis=1)
 coords=[]
 for j,s in [(13,11),(14,12),(15,11),(16,12)]:
  d=(seq[:,j]-seq[:,s])/np.maximum(scale[:,None],1);coords.extend([(d*side).sum(1),(d*up).sum(1)])
 wrist_above=[]
 for w,e in [(15,13),(16,14)]:wrist_above.append(((seq[:,w]-seq[:,e])*up).sum(1)/np.maximum(scale,1))
 curves=[le,re,lb,rb,abs(le-re),abs(lb-rb),*coords,*wrist_above]
 # Distances to arm prototypes are geometric inputs, not target or rule scores.
 distances=[abs(le-.75)+abs(re-.75)+(1-lb)+(1-rb),abs(le-.5)+abs(re-.5)+(1-lb)+(1-rb),abs(le-.5)+abs(re-.5)+abs(lb-.5)+abs(rb-.5)+np.maximum(0,-wrist_above[0])+np.maximum(0,-wrist_above[1])]
 vector=[]
 for a in curves+distances:
  q=a[valid];vector.extend(np.percentile(q,[5,25,50,75,95]).tolist())
 for a in distances:
  windows=[np.mean(a[i:i+3]) for i in range(len(a)-2) if valid[i:i+3].all()];vector.append(float(min(windows)) if windows else float('nan'))
 return vector if np.isfinite(vector).all() else None

def relative_knee_gap(seq):
 valid=np.isfinite(seq[:,[23,24,25,26,27,28]]).all((1,2))
 if len(seq)<10 or valid.mean()<.8 or valid[:3].sum()<2:return None
 gap=abs(seq[:,25,0]-seq[:,26,0])/np.maximum(abs(seq[:,27,0]-seq[:,28,0]),1)
 initial=float(np.median(gap[:3][valid[:3]]));return np.r_[np.percentile(gap[valid],[5,10,25,50,75,90,95]),np.percentile((gap-initial)[valid],[5,10,25,50,75,90,95]),initial].tolist()

def relative_trunk(seq,reference=None):
 valid=np.isfinite(seq[:,[11,12,23,24]]).all((1,2))
 if len(seq)<10 or valid.mean()<.8 or valid[:3].sum()<2:return None
 sh=seq[:,[11,12]].mean(1);hip=seq[:,[23,24]].mean(1);tilt=np.degrees(np.arctan2(sh[:,0]-hip[:,0],hip[:,1]-sh[:,1]));initial=float(np.median(tilt[:3][valid[:3]]))
 if reference is not None:
  ref=np.asarray(reference);good=np.isfinite(ref[:,[11,12,23,24]]).all((1,2))
  if good.sum()<2:return None
  s=ref[:,[11,12]].mean(1);h=ref[:,[23,24]].mean(1);initial=float(np.median(np.degrees(np.arctan2(s[good,0]-h[good,0],h[good,1]-s[good,1]))))
 change=abs(tilt-initial)
 return np.r_[np.percentile(change[valid],[5,10,25,50,75,90,95]),np.percentile(abs(tilt[valid]),[5,50,95])].tolist()
