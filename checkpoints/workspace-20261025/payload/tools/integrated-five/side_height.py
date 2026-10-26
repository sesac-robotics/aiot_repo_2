"""Fixed feet distinguish sagittal hip travel from whole-image translation."""
import numpy as np

def shoulder_phase(x,ref):
    """A hinge attempt can move the shoulders while hip height stays constant.

    Use that observed excursion only to find the return window. All six
    measurements still use actual hip/shoulder coordinates, never filled joints.
    """
    valid=np.isfinite(x[:,[11,12,23,24]]).all((1,2))
    if valid.mean()<.8 or valid.sum()<10 or valid[-3:].sum()<2:return None
    hip=x[:,[23,24]].mean(1);shoulder=x[:,[11,12]].mean(1);scale=ref['scale']
    drop=(hip[:,1]-ref['hip_y'])/scale
    if np.nanmax(drop)>=.08:return None
    shoulder_drop=(shoulder[:,1]-ref['shoulder_y'])/scale
    if np.nanmax(shoulder_drop)<.08:return None
    ratios=np.linalg.norm(hip[valid]-shoulder[valid],axis=1)/scale
    if np.any((ratios<.4)|(ratios>2.5)):return None
    bottom=int(np.nanargmax(shoulder_drop));tail_start=max(bottom+1,len(x)-max(3,len(x)//4))
    if tail_start>len(x)-3:return None
    windows=[(abs(float(np.median(shoulder_drop[i:i+3]))),i) for i in range(tail_start,len(x)-2) if valid[i:i+3].all()]
    if not windows:return None
    _,top=min(windows);end=float(np.median(drop[top:top+3]));tail=drop[tail_start:];tail=tail[np.isfinite(tail)];torso=shoulder[top:top+3]-hip[top:top+3]
    return dict(x=[abs(end),abs(end)/max(.08,float(np.nanmax(drop))),abs(float(np.median(tail))),abs(float(np.median(shoulder[top:top+3,1]))-ref['shoulder_y'])/scale,float(np.median(np.linalg.norm(torso,axis=1)))/scale,abs(float(np.median(torso[:,0]))-ref.get('torso_x',0.))/scale],scalar=abs(end),coverage=float(valid.mean()),baseline=ref,bottom_index=bottom,return_index=top,peak_drop=float(np.nanmax(drop)),phase_source='observed_shoulder_excursion_with_insufficient_hip_descent',feature_version='standing-return-height-v4-side-v2')

def reference(xy, height_features):
    x=np.asarray(xy,float)
    ref=height_features.reference(x)
    if ref is None:return None
    valid=np.isfinite(x[:,[27,28]]).all((1,2))
    if valid.sum()<3:return ref
    feet=x[valid][:,[27,28]].mean(1)
    ref.update(foot_x=float(np.median(feet[:,0])),foot_y=float(np.median(feet[:,1])))
    return ref

def extract(xy, ref, width, height, height_features):
    x=np.asarray(xy,float).copy()
    if ref is None:return None
    if x.ndim!=3 or x.shape[1:]!=(33,2) or len(x)<10:return None
    if width is not None and height is not None:
        x[(x[:,:,0]<0)|(x[:,:,0]>=width)|(x[:,:,1]<0)|(x[:,:,1]>=height)]=np.nan
    if not np.isfinite(ref['scale']) or ref['scale']<1:return None
    valid=np.isfinite(x[:,[23,24,27,28]]).all((1,2))
    feet=x[:,[27,28]].mean(1)
    feet_observed=all(k in ref for k in ['foot_x','foot_y']) and valid.mean()>=.8
    # These are observation guards, not return score thresholds.
    if feet_observed:
        if np.max(abs(feet[valid,0]-ref['foot_x']))/ref['scale']>.2:return None
        if np.max(abs(feet[valid,1]-ref['foot_y']))/ref['scale']>.2:return None
    # Hip travel during a side squat is expected. Cancel only its horizontal
    # translation after checking fixed feet. Vertical hip/shoulder heights and
    # torso inclination remain unchanged; never center the vertical channel.
    hip=x[:,[23,24]].mean(1)
    x[:,:,0]-=(hip[:,0]-ref['center_x'])[:,None]
    result=height_features.extract(x,ref,None,None)
    if result is None:result=shoulder_phase(x,ref)
    if result is None:return None
    top=result['return_index']
    # When feet are hidden, retain the old .45 torso-length location bound
    # at the actual return, rather than rejecting expected mid-squat hip travel.
    if not feet_observed and np.max(abs(hip[top:top+3,0]-ref['center_x']))/ref['scale']>.45:return None
    return result
