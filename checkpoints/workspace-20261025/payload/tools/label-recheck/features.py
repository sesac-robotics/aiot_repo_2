"""Compact geometric measurements, without scores or target labels."""
import sys
from pathlib import Path
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');sys.path.insert(0,str(B/'five-exercise-comparison'));import geometry

def arm_curves(seq):
 valid=np.isfinite(seq[:,[11,12,13,14,15,16,23,24]]).all((1,2))
 if len(seq)<10 or valid.mean()<.8:return None
 ch=geometry.channels(seq);a=np.array([ch[k] for k in ['left_elevation_deg','right_elevation_deg','left_elbow_deg','right_elbow_deg']]).T
 sh=seq[:,[11,12]].mean(1);hip=seq[:,[23,24]].mean(1);up=sh-hip;scale=np.linalg.norm(up,axis=1);valid&=scale>1;up=up/np.maximum(scale[:,None],1)
 wrists=np.array([((seq[:,w]-seq[:,e])*up).sum(1)/np.maximum(scale,1) for w,e in [(15,13),(16,14)]]).T
 valid&=np.isfinite(a).all(1)&np.isfinite(wrists).all(1)
 return (a,wrists,valid) if valid.mean()>=.8 else None

def shape(seq,kind):
 result=arm_curves(seq)
 if result is None:return None
 a,w,valid=result;le,re,lb,rb=a.T
 if kind in ['Y','T','press']:
  target={'Y':135.,'T':90.,'press':180.}[kind]
  deviations=np.array([abs(le-target),abs(re-target),180-lb,180-rb,abs(le-re),abs(lb-rb)]).T/180
 else:
  # W has bent elbows and raised forearms; upper arms can be below horizontal.
  upper=np.minimum.reduce([abs(le-t)+abs(re-t) for t in [45.,60.,90.]])/360
  deviations=np.array([upper,abs(lb-90)/180,abs(rb-90)/180,abs(le-re)/180,abs(lb-rb)/180,np.maximum(0,-w[:,0])+np.maximum(0,-w[:,1])]).T
 cost=deviations.mean(1);windows=[i for i in range(len(seq)-2) if valid[i:i+3].all()]
 if not windows:return None
 best=min(windows,key=lambda i:float(cost[i:i+3].mean()));local=slice(best,best+3)
 return np.r_[np.median(deviations[local],0),np.median(a[local],0)/180,np.median(w[local],0),np.percentile(cost[valid],[5,25,50]),cost[local].mean(),best/max(1,len(seq)-3)].tolist()

def bilateral(seq):
 result=arm_curves(seq)
 if result is None:return None
 a,w,v=result;le,re,lb,rb=a.T
 gap=abs(le-re)/180;elbowgap=abs(lb-rb)/180
 curves=[gap[v],elbowgap[v],np.minimum(le,re)[v]/180,np.maximum(le,re)[v]/180]
 return np.r_[*[np.percentile(c,[10,50,90]) for c in curves],abs(np.argmax(np.where(v,le,-1))-np.argmax(np.where(v,re,-1)))/len(seq)].tolist()

def side_body(seq):
 options=[]
 for s,h,k,a in [(11,23,25,27),(12,24,26,28)]:
  valid=np.isfinite(seq[:,[s,h,k,a]]).all((1,2))
  if len(seq)<10 or valid.mean()<.8:continue
  start=seq[:,s];end=seq[:,a];axis=end-start;length=np.linalg.norm(axis,axis=1);v=valid&(length>1)
  if v.mean()<.8:continue
  offset=seq[:,h]-start;cross=axis[:,0]*offset[:,1]-axis[:,1]*offset[:,0];distance=abs(cross)/np.maximum(length**2,1)
  def angle(u,c,w):
   x=seq[:,u]-seq[:,c];y=seq[:,w]-seq[:,c];return np.degrees(np.arccos(np.clip((x*y).sum(1)/np.maximum(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1),1),-1,1)))
  hip_deficit=(180-angle(s,h,a))/180;knee_deficit=(180-angle(h,k,a))/180
  feature=np.r_[np.percentile(distance[v],[5,25,50,75,95]),np.percentile(hip_deficit[v],[5,25,50,75,95]),np.percentile(knee_deficit[v],[10,50,90])]
  if np.isfinite(feature).all():options.append((float(v.mean()),feature))
 if not options:return None
 # Pick the most observed side, never the side that yields a better posture score.
 return max(options,key=lambda t:t[0])[1].tolist()
