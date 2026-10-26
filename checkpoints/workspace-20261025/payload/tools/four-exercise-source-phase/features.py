"""Source-clock observations only; missing tracking never becomes a severity label."""
import sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
B=Path('/home/jayden/SeSACPAI-Mid')
sys.path[:0]=[str(B/'tools/label-recheck'),str(B/'tools/return-label-expansion-v3')]
import importlib.util
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
compact=module('source_phase_compact',B/'tools/label-recheck/features.py')
cycle=module('source_phase_cycle',B/'tools/return-label-expansion-v3/cycle_features.py')

def sanitize(xy,confidence,width,height):
 x=np.asarray(xy,float).copy();q=np.asarray(confidence,float)
 ok=np.isfinite(x).all(2)&np.isfinite(q).all(2)&(q.min(2)>=.5)&(x[:,:,0]>=0)&(x[:,:,0]<width)&(x[:,:,1]>=0)&(x[:,:,1]<height)
 x[~ok]=np.nan
 return x

def continuous_windows(valid,frames,fps,n):
 return [i for i in range(len(valid)-n+1) if valid[i:i+n].all() and np.all(np.diff(frames[i:i+n])/fps<=.16) and (frames[i+n-1]-frames[i])/fps>=(n-1)*.06]

def stage(seq,frames,fps,kind):
 """Best five consecutive observations of the requested shape, never clip-tail thirds.

 Geometry prototype deviations are unscored features. No label, grade, filename,
 person, clip position or provider classification is part of the vector.
 """
 r=compact.arm_curves(seq)
 if r is None:return dict(status='held',reason='required_arm_and_torso_observation_below_80_percent')
 a,w,v=r;le,re,lb,rb=a.T
 if kind in ('Y','T'):
  t={'Y':135.,'T':90.}[kind];dev=np.array([abs(le-t),abs(re-t),180-lb,180-rb,abs(le-re),abs(lb-rb)]).T/180
 else:
  upper=np.minimum.reduce([abs(le-t)+abs(re-t) for t in (45.,60.,90.)])/360
  dev=np.array([upper,abs(lb-90)/180,abs(rb-90)/180,abs(le-re)/180,abs(lb-rb)/180,np.maximum(0,-w[:,0])+np.maximum(0,-w[:,1])]).T
 cost=dev.mean(1);starts=continuous_windows(v,np.asarray(frames),fps,5)
 if not starts:return dict(status='held',reason='five_contiguous_observations_not_available')
 start=min(starts,key=lambda i:float(cost[i:i+5].mean()));ix=np.arange(start,start+5)
 f=np.r_[np.median(dev[ix],0),np.median(a[ix],0)/180,np.median(w[ix],0),np.percentile(cost[v],[5,25,50]),cost[ix].mean(),np.std(a[ix],0)/180]
 assert len(f)==20 and np.isfinite(f).all()
 return dict(status='features_ready',feature=f.tolist(),feature_version='ytw-stage-five-contiguous-v4',stage=kind,window_source_frames=np.asarray(frames)[ix].astype(int).tolist(),observed_fraction=float(v.mean()),available_at_source_frame=int(frames[-1]),selection_uses_complete_supplied_attempt=True,order_quality_inferred=False)

def return_turn(seq,confidence,frames,fps,core_end):
 """Completed observed arm excursion. No extra post-window frames are supplied."""
 frames=np.asarray(frames)
 if len(frames)<8 or np.any(np.diff(frames)/fps>.16):return dict(status='held',reason='short_or_gapped_source_observation')
 r=cycle.extract(seq,confidence,frames,core_end)
 if r is None:return dict(status='held',reason='closed_post_descent_return_or_required_arm_unobserved')
 r.update(status='features_ready',available_at_source_frame=int(frames[-1]),future_after_supplied_window_used=False,post_window_context_seconds=0.)
 return r
