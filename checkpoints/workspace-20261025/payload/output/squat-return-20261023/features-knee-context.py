"""Height return features. No head/ankle gate, labels or provider hints as inputs."""
import numpy as np
VERSION='standing-return-height-v5-knee-context'
NAMES=['기립 기준 대비 복귀 골반 높이 차','하강량 대비 남은 높이 차','복귀 끝부분 골반 높이 차 중앙값','기립 기준 대비 복귀 어깨 높이 차','복귀 몸통 길이 비율','시작 대비 복귀 몸통 기울기 변화']

def reference(xy):
 x=np.asarray(xy,float);valid=np.isfinite(x[:,[11,12,23,24]]).all((1,2));ix=np.flatnonzero(valid[:8])
 if len(ix)<3:return None
 t=x[ix[:3]];s=t[:,[11,12]].mean(1);h=t[:,[23,24]].mean(1);scale=float(np.median(np.linalg.norm(h-s,axis=1)))
 if scale<1:return None
 ref=dict(hip_y=float(np.median(h[:,1])),shoulder_y=float(np.median(s[:,1])),center_x=float(np.median(h[:,0])),scale=scale,torso_x=float(np.median((s-h)[:,0])))
 if np.isfinite(t[:,[25,26]]).all():
  for side in [0,1]:
   delta=t[:,25+side]-t[:,23+side];ref['knee_gap_'+str(side)]=float(np.median(delta[:,1]))/scale;ref['thigh_'+str(side)]=float(np.median(np.linalg.norm(delta,axis=1)))/scale
 return ref

def extract(xy,baseline=None,width=None,height=None):
 x=np.asarray(xy,float).copy()
 if x.ndim!=3 or x.shape[1:]!=(33,2) or len(x)<10:return None
 if width is not None and height is not None:
  x[(x[:,:,0]<0)|(x[:,:,0]>=width)|(x[:,:,1]<0)|(x[:,:,1]>=height)]=np.nan
 valid=np.isfinite(x[:,[11,12,23,24]]).all((1,2));ref=baseline or reference(x)
 if ref is None or valid.mean()<.8 or valid.sum()<10 or valid[-3:].sum()<2:return None
 scale=ref['scale']
 if not np.isfinite(list(ref.values())).all() or scale<1:return None
 h=x[:,[23,24]].mean(1);s=x[:,[11,12]].mean(1);drop=(h[:,1]-ref['hip_y'])/scale
 # Never use future observations to establish a reference. Movement/scale
 # changes make the previous standing location unusable.
 if np.nanmax(abs(h[valid,0]-ref['center_x']))/scale>.45:return None
 ratios=np.linalg.norm(h[valid]-s[valid],axis=1)/scale
 if np.any((ratios<.4)|(ratios>2.5)):return None
 bottom=int(np.nanargmax(drop));peak=float(drop[bottom]);tail_start=max(bottom+1,len(x)-max(3,len(x)//4))
 if peak<.08 or tail_start>len(x)-3:return None
 windows=[(float(np.median(drop[i:i+3])),i) for i in range(tail_start,len(x)-2) if valid[i:i+3].all()]
 if not windows:return None
 end,top=min(windows);tail=drop[tail_start:];tail=tail[np.isfinite(tail)];median=float(np.median(tail))
 shoulder_residual=float(np.median(s[top:top+3,1])-ref['shoulder_y'])/scale
 torso=s[top:top+3]-h[top:top+3];torso_ratio=float(np.median(np.linalg.norm(torso,axis=1)))/scale;tilt=abs(float(np.median(torso[:,0]))-ref.get('torso_x',0.))/scale
 context=None
 if all('knee_gap_'+str(side) in ref for side in [0,1]) and np.isfinite(x[top:top+3,[25,26]]).all():
  context=[]
  for side in [0,1]:
   delta=x[top:top+3,25+side]-x[top:top+3,23+side];context.extend([ref['knee_gap_'+str(side)]-float(np.median(delta[:,1]))/scale,ref['thigh_'+str(side)]-float(np.median(np.linalg.norm(delta,axis=1)))/scale])
 return dict(x=[abs(end),abs(end)/max(.08,peak),abs(median),abs(shoulder_residual),torso_ratio,tilt],optional_knee_context=context,scalar=abs(end),coverage=float(valid.mean()),baseline=ref,bottom_index=bottom,return_index=top,peak_drop=peak,feature_version=VERSION)
