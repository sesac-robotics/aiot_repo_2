"""Observe the first completed elbow excursion, never the arbitrary clip tail."""
import numpy as np
VERSION='observed-first-elbow-return-v1'
def locate(angles,core_frames):
 a=np.asarray(angles,float);core=np.asarray(core_frames,bool)
 if len(a)<8 or len(a)!=len(core):return None
 # Short median smoothing is phase-only; final quantiles use original angles.
 s=np.array([np.nanmedian(a[max(0,i-1):min(len(a),i+2)]) for i in range(len(a))]);state='ready';high=s[0];bottom=0;low=s[0];top=None;best=None;confirmed=False
 for i,v in enumerate(s):
  if not np.isfinite(v):continue
  if state=='ready':
   high=max(high,v)
   if high-v>=15:state='descending';low=v;bottom=i
  elif state=='descending':
   if v<low:low=v;bottom=i
   if v-low>=15:
    if not core[bottom]:return None
    state='ascending';best=v;top=i
  else:
   if v>best:best=v;top=i
   if best-v>=12:confirmed=True;break
 if state!='ascending' or top is None:return None
 # EOF alone never proves recovery; accept only a stable observed upper hold
 # or an observed next descent, never an ascent cut in progress.
 if not confirmed:
  tail=s[top:]
  if len(tail)<3 or np.ptp(tail)>4:return None
 peak=max(range(max(bottom+1,top-1),min(len(a),top+2)),key=lambda j:a[j] if np.isfinite(a[j]) else -np.inf)
 if peak<=bottom:return None
 return dict(bottom_index=int(bottom),return_index=int(peak),next_descent_observed=confirmed,phase_end_index=int(i),smoothing='three_sample_median_phase_only')
def extract(xy,confidence,frames,core_end_frame):
 x=np.asarray(xy,float);q=np.asarray(confidence,float);frames=np.asarray(frames);choices=[]
 for side,js in [('left',[11,13,15]),('right',[12,14,16])]:
  ok=np.isfinite(x[:,js]).all((1,2))
  if len(x)<8 or ok.mean()<.8:continue
  u=x[:,js[0]]-x[:,js[1]];v=x[:,js[2]]-x[:,js[1]];den=np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1);valid=ok&(den>1)
  angles=np.full(len(x),np.nan);angles[valid]=np.degrees(np.arccos(np.clip((u[valid]*v[valid]).sum(1)/den[valid],-1,1)))
  phase=locate(angles,frames<core_end_frame)
  if phase is None:continue
  top=phase['return_index'];ids=np.arange(max(phase['bottom_index']+1,top-1),min(len(x),top+2));ids=ids[valid[ids]]
  if len(ids)<2:continue
  f=np.r_[np.percentile(angles[ids],[5,10,25,50,75,90,95]),valid.mean()].tolist();reliability=float(np.minimum(q[:,js,0],q[:,js,1])[valid].mean());choices.append((reliability,dict(feature=f,side=side,feature_version=VERSION,return_source_frame=int(frames[top]),bottom_source_frame=int(frames[phase['bottom_index']]),phase=phase,window_source_frames=frames[ids].astype(int).tolist())))
 return max(choices,key=lambda x:x[0])[1] if choices else None
