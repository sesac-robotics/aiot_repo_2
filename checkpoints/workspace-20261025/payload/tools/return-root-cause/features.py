"""Observed post-bottom upper reversal, without the last-quarter timing bias."""
import numpy as np
VERSION='height-observed-upper-reversal-v1'

def with_reference_pose(xy,reference_xy,base):
    if base is None:return None
    x=np.asarray(xy,float);r=np.asarray(reference_xy,float);top=base['return_index'];candidates=[]
    for chain in [[11,23,25,27],[12,24,26,28]]:
        valid=np.isfinite(x[:,chain]).all((1,2));ready=np.isfinite(r[:,chain]).all((1,2))
        if valid.mean()>=.8 and valid[top:top+3].all() and ready.sum()>=3:candidates.append((float(valid.mean()),chain,ready))
    if not candidates:return None
    _,chain,ready=max(candidates,key=lambda a:a[0]);q=np.median(x[top:top+3,chain],axis=0);p=np.median(r[ready][:,chain],axis=0)
    leg=lambda a:float(np.linalg.norm(a[1]-a[2])+np.linalg.norm(a[2]-a[3]))
    a,b=leg(p),leg(q)
    if min(a,b)<1 or not .7<=b/a<=1.3:return None
    # Coordinates relative to the observed support, normalized by projected
    # leg length. This retains hip/shoulder height while reducing scale drift.
    initial=(p-p[3])/a;returned=(q-q[3])/b;delta=returned-initial
    return dict(x=base['x']+delta.flatten().tolist()+[b/a],side='left' if chain[0]==11 else 'right',feature_version='height-initial-reference-pose-v1',return_index=top,bottom_index=base['bottom_index'])

def extract(xy,ref,width=None,height=None):
    x=np.asarray(xy,float).copy()
    if ref is None or x.ndim!=3 or x.shape[1:]!=(33,2) or len(x)<10:return None
    if width is not None and height is not None:
        x[(x[:,:,0]<0)|(x[:,:,0]>=width)|(x[:,:,1]<0)|(x[:,:,1]>=height)]=np.nan
    valid=np.isfinite(x[:,[11,12,23,24]]).all((1,2))
    if valid.mean()<.8 or valid.sum()<10:return None
    scale=ref['scale']
    if not np.isfinite(list(ref.values())).all() or scale<1:return None
    hip=x[:,[23,24]].mean(1);shoulder=x[:,[11,12]].mean(1)
    hd=(hip[:,1]-ref['hip_y'])/scale;sd=(shoulder[:,1]-ref['shoulder_y'])/scale
    ratios=np.linalg.norm(hip[valid]-shoulder[valid],axis=1)/scale
    if np.any((ratios<.4)|(ratios>2.5)):return None
    phase=(hd+sd)/2
    windows=[(float(np.median(phase[i:i+3])),i) for i in range(len(x)-2) if valid[i:i+3].all()]
    if not windows:return None
    # Confirm the first lower reversal by an observed ascent. A deeper, clipped
    # next descent must not replace the bottom of the completed first attempt.
    peak,bottom=windows[0];confirmed=False
    for value,i in windows[1:]:
        if value>peak:peak,bottom=value,i
        if peak>=.08 and value<=peak-.08:confirmed=True;break
    if not confirmed:return None
    after=[(value,i) for value,i in windows if i>=bottom+3]
    if not after:return None
    low,top=after[0]
    for value,i in after[1:]:
        if value<low:low,top=value,i
        elif value>=low+.08:break
    if np.max(abs(hip[top:top+3,0]-ref['center_x']))/scale>.45:return None
    end=float(np.median(hd[top:top+3]));s_end=float(np.median(sd[top:top+3]));torso=shoulder[top:top+3]-hip[top:top+3]
    around=hd[max(bottom+3,top-1):min(len(x),top+4)];around=around[np.isfinite(around)]
    return dict(x=[abs(end),abs(end)/max(.08,float(np.nanmax(hd))),abs(float(np.median(around))),abs(s_end),float(np.median(np.linalg.norm(torso,axis=1)))/scale,abs(float(np.median(torso[:,0]))-ref.get('torso_x',0.))/scale],return_index=top,bottom_index=bottom,feature_version=VERSION,coverage=float(valid.mean()))
