"""Observed anatomical lower-turn features; no inferred grades or ready baseline.

Event native frame windows must come from direct source review (training) or a
separately validated event detector (live). Passing this extractor is not tracking QA.
"""
import numpy as np
from observation_features_v4 import dense_samples,quality
from phase_features_v3 import angle,JOINTS
VERSION='shoulder-anatomical-lower-turn-v1'
CRITERION='observed_anatomical_lower_turn'
NAMES=['left_elbow_turn_deg','right_elbow_turn_deg',
       'left_upper_arm_turn_elevation_deg','right_upper_arm_turn_elevation_deg',
       'left_wrist_turn_height_fixed_event_torso','right_wrist_turn_height_fixed_event_torso']

def extract(samples,event,source_fps,width=640,height=480,view='front',native_frame_numbers=None):
    def hold(reason,**kw):return dict(status='held',reason=reason,version=VERSION,criterion=CRITERION,**kw)
    if view!='front':return hold('frontal_target_required')
    if not np.isfinite(source_fps) or source_fps<=0:raise ValueError('source frame clock required')
    ticks,x=dense_samples(samples,width,height)
    if not len(ticks):return hold('empty_input')
    if event.get('protocol')!='shoulder_press' or event.get('laterality')!='bilateral':return hold('reviewed_bilateral_press_event_required')
    if event.get('posture') not in ('standing','seated'):return hold('posture_bank_unknown')
    names=['peak_frames','turn_frames']
    closure=event.get('closure')
    if closure=='observed_next_ascent':names.append('next_ascent_frames')
    elif closure!='observed_rack' or event.get('rack_visible_stable') is not True:return hold('closed_event_required')
    windows=[]
    for name in names:
        w=event.get(name)
        if not isinstance(w,list) or len(w)!=2 or any(type(v) is not int or v<0 for v in w) or w[1]<w[0]:return hold('native_frame_windows_required')
        windows.append(w)
    if any(a[1]>=b[0] for a,b in zip(windows,windows[1:])):return hold('peak_turn_next_ascent_order_invalid')
    # Actual native frame numbers accompany the source-pixel samples. The 10Hz
    # slot is an observation clock, not a replacement for a 15fps native timestamp.
    if native_frame_numbers is None or len(native_frame_numbers)!=len(samples):return hold('actual_native_frame_alignment_required')
    native=np.asarray(native_frame_numbers)
    if not np.issubdtype(native.dtype,np.integer) or np.any(np.diff(native)<=0):raise ValueError('strictly increasing native frames required')
    frames=np.full(len(ticks),np.nan);frames[np.array([t for t,_ in samples])-ticks[0]]=native
    masks=[(frames>=a)&(frames<=b) for a,b in windows]
    if any(m.sum()<3 for m in masks):return hold('three_actual_event_slots_required')
    first,last=np.flatnonzero(masks[0])[0],np.flatnonzero(masks[-1])[-1]
    context=(ticks>=ticks[first])&(ticks<=ticks[last])
    if (frames[first]-windows[0][0])/source_fps>.15 or (windows[-1][1]-frames[last])/source_fps>.15:
        return hold('event_context_not_fully_supplied')
    valid=np.isfinite(x[:,JOINTS]).all((1,2))
    coverage=float(valid[context].mean()) if context.any() else 0.
    if not context.any() or coverage<.8:return hold('insufficient_actual_slot_observation',coverage=coverage)
    # Event continuity is stricter than average coverage; gaps are not descent evidence.
    if not valid[context].all():return hold('missing_joint_inside_event_context',coverage=coverage)
    if any(m.sum()<3 or not valid[m].all() for m in masks):return hold('three_actual_event_slots_required',coverage=coverage)
    q=quality(ticks[context],x[context])
    if not q['passed']:return hold('tracking_geometry_uncertain',coverage=coverage,quality=q)
    elev=np.stack([angle(x,23,11,13),angle(x,24,12,14)],1)
    peak=np.median(elev[masks[0]],0);turn=np.median(elev[masks[1]],0)
    if not np.any(peak-turn>=15):return hold('observed_descent_not_established',coverage=coverage,quality=q)
    if closure=='observed_next_ascent' and not np.any(np.median(elev[masks[2]],0)-turn>=8):
        return hold('observed_next_ascent_not_established',coverage=coverage,quality=q)
    sh=x[:,[11,12]].mean(1);hip=x[:,[23,24]].mean(1)
    scale=float(np.median(np.linalg.norm(sh[context]-hip[context],axis=1)))
    if scale<1:return hold('degenerate_event_torso')
    elbows=np.stack([angle(x,11,13,15),angle(x,12,14,16)],1)
    wrist=(x[:,[11,12],1]-x[:,[15,16],1])/scale
    f=np.r_[np.median(elbows[masks[1]],0),turn,np.median(wrist[masks[1]],0)]
    return dict(status='features_ready',version=VERSION,criterion=CRITERION,
        bank=f'{VERSION}/{view}/{event["posture"]}/bilateral',feature_names=NAMES,features=f.tolist(),
        observed_event_ticks={name:ticks[m].tolist() for name,m in zip(names,masks)},
        fixed_event_torso_scale=scale,coverage=coverage,quality=q,
        ready_reference_used=False,grade=None,score=None,label_inferred=False,
        tracking_QA_proven=False,production_approved=False)
