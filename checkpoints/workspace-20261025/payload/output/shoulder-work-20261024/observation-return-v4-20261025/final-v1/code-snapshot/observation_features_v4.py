"""Offline candidate. Tracking plausibility is never a posture grade.

Only frontal geometry is supported. No interpolation, camera access or training.
"""
import numpy as np
from phase_features_v3 import extract as phase_v3

RETURN_SCHEMA='shoulder-ready-return-relative-quality-v4'
BODY_SCHEMA='shoulder-frontal-four-joint-window-quality-v2'
QUALITY_SCHEMA='shoulder-frontal-topology-screen-v1'
BODY_JOINTS=[11,12,23,24]
BODY_NAMES=['absolute_tilt_p50_deg','absolute_tilt_p90_deg','absolute_tilt_p95_deg',
            'window_initial_relative_tilt_p50_deg','window_initial_relative_tilt_p90_deg','window_initial_relative_tilt_p95_deg']

def dense_samples(samples,width,height):
    if not samples:return np.array([],int),np.empty((0,33,2))
    ticks=np.asarray([t for t,_ in samples])
    if not np.issubdtype(ticks.dtype,np.integer) or np.any(np.diff(ticks)<=0):
        raise ValueError('strictly increasing integer 10 Hz slots required')
    points=np.asarray([p for _,p in samples],float)
    if points.shape!=(len(ticks),33,2):raise ValueError('expected source-pixel 33x2 joints')
    x=np.full((int(ticks[-1]-ticks[0]+1),33,2),np.nan)
    x[ticks-ticks[0]]=points
    wh=np.array([width,height],float)
    if not np.isfinite(wh).all() or np.any(wh<=0):raise ValueError('positive dimensions required')
    bad=~np.isfinite(x).all(2)|((x<wh*.02)|(x>wh*.98)).any(2)
    x[bad]=np.nan
    return np.arange(ticks[0],ticks[-1]+1),x

def quality(ticks,x):
    """Conservative engineering screen, not an anatomical correctness guarantee.

    Check severe projected torso collapse and asymmetric width collapse only.
    Tilt is deliberately not a quality threshold: actual leaning must remain measurable.
    This screen cannot identify all plausible but misplaced shoulders.
    """
    valid=np.isfinite(x[:,BODY_JOINTS]).all((1,2))
    sh=x[:,[11,12]].mean(1);hip=x[:,[23,24]].mean(1)
    axis=np.linalg.norm(sh-hip,axis=1)
    sw=np.linalg.norm(x[:,11]-x[:,12],axis=1)
    hw=np.linalg.norm(x[:,23]-x[:,24],axis=1)
    # A frontal width <12% of trunk axis is too poorly conditioned for this target.
    collapse=valid&((np.minimum(sw,hw)<axis*.12)|(axis<1))
    disproportion=valid&(np.maximum(sw,hw)>4*np.minimum(sw,hw))
    flags=[]
    for i in np.flatnonzero(collapse|disproportion):
        flags.append({'tick':int(ticks[i]),'reasons':
                      (['projected_torso_width_collapsed'] if collapse[i] else [])+
                      (['projected_torso_width_disproportion'] if disproportion[i] else [])})
    return {'schema':QUALITY_SCHEMA,'flags':flags,'passed':not flags,
            'semantics':'tracking_or_view_uncertainty_not_posture_error',
            'anatomical_correctness_guaranteed':False}

def extract_return(samples,width=640,height=480,view='front'):
    ticks,x=dense_samples(samples,width,height)
    if not len(ticks):return dict(status='held',reason='empty_input',version=RETURN_SCHEMA)
    if view!='front':return dict(status='held',reason='frontal_geometry_required',version=RETURN_SCHEMA)
    result=phase_v3(list(zip(ticks,x)),width,height)
    # Quality must cover the selected first attempt, not a later repetition.
    if result['status']=='features_ready':
        selected=(ticks>=result['reference_ticks'][0])&(ticks<=result['return_ticks'][-1])
        q=quality(ticks[selected],x[selected])
        if not q['passed']:
            return dict(status='held',reason='tracking_geometry_uncertain',version=RETURN_SCHEMA,quality=q)
        result['quality']=q
    result.update(version=RETURN_SCHEMA)
    # This API does not expose the old eight-joint trunk features as a new body model.
    result.pop('trunk_features',None);result.pop('trunk_feature_names',None)
    return result

def extract_body(samples,window,exercise_confirmed=False,width=640,height=480,view='front'):
    """Explicit observed exercise window. Its first 5 slots are NOT a ready state.

    Arms, head and ankles are irrelevant. Caller must confirm the exercise attempt;
    arbitrary waiting footage cannot silently become an exercise or training sample.
    """
    def hold(reason,**kw):return dict(status='held',reason=reason,version=BODY_SCHEMA,**kw)
    if not exercise_confirmed:return hold('exercise_window_not_confirmed')
    if view!='front':return hold('frontal_geometry_required')
    if window is None or len(window)!=2 or any(not isinstance(t,(int,np.integer)) for t in window) or window[1]<window[0]:
        return hold('explicit_integer_window_required')
    ticks,x=dense_samples(samples,width,height)
    if not len(ticks) or ticks[0]>window[0] or ticks[-1]<window[1]:return hold('window_not_fully_supplied')
    mask=(ticks>=window[0])&(ticks<=window[1]);ticks=ticks[mask];x=x[mask]
    valid=np.isfinite(x[:,BODY_JOINTS]).all((1,2));coverage=float(valid.mean())
    if len(ticks)<8 or valid.sum()<8 or coverage<.8:return hold('insufficient_actual_slot_observation',coverage=coverage)
    q=quality(ticks,x)
    if not q['passed']:return hold('tracking_geometry_uncertain',coverage=coverage,quality=q)
    if not valid[:5].all():return hold('window_initial_reference_unobservable',coverage=coverage,quality=q)
    sh=x[:,[11,12]].mean(1);hip=x[:,[23,24]].mean(1)
    tilt=np.degrees(np.arctan2(sh[:,0]-hip[:,0],hip[:,1]-sh[:,1]))
    baseline=float(np.median(tilt[:5]));v=tilt[valid]
    f=np.r_[np.percentile(abs(v),[50,90,95]),np.percentile(abs(v-baseline),[50,90,95])]
    return dict(status='features_ready',version=BODY_SCHEMA,quality=q,coverage=coverage,
                feature_names=BODY_NAMES,features=f.tolist(),window_ticks=list(map(int,window)),
                observed_ticks=ticks[valid].tolist(),baseline_ticks=ticks[:5].tolist(),
                baseline_semantics='first_five_actual_window_slots_not_initial_ready',
                grade=None,score=None,label_inferred=False)
