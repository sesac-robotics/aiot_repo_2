"""Proposal-only automatic attempt evidence; deliberately not scoring approval."""
import hashlib
import numpy as np
VERSION='ytw-auto-attempt-evidence-v1-proposal'
PHASES=('Y','T','W')

def create(event,*,state_model_sha256,source_sha256,protocol='upright_Y_T_W_v1'):
    samples=event['samples'];start,end=event['start_tick'],event['end_tick']
    ticks=np.array([t for t,p in samples]);xy=np.array([p for t,p in samples],dtype='<f8')
    if not np.array_equal(ticks,np.arange(start,end+1)) or xy.shape!=(end-start+1,33,2):
        raise ValueError('Dense actual 10Hz slots required')
    width,height=event['width'],event['height']
    if width<=0 or height<=0:
        raise ValueError('Positive image dimensions required')
    finite=np.isfinite(xy).all(2)
    if (((xy<np.array([width,height])*.02)|(xy>np.array([width,height])*.98)).any(2)&finite).any():
        raise ValueError('Border-unsafe coordinates in automatic evidence')
    windows=event['auto_phase_windows']
    if set(windows)!=set(PHASES) or any(not 0<=windows[p][0]<=windows[p][1]<len(xy) for p in PHASES):
        raise ValueError('Three ordered automatic windows required')
    if not (windows['Y'][1]<windows['T'][0] and windows['T'][1]<windows['W'][0]):
        raise ValueError('Overlapping or reversed phases')
    if protocol!='upright_Y_T_W_v1' or not event['ordered_attempt_complete']:
        raise ValueError('Supported ordered attempt required')
    required=[11,12,13,14,15,16,23,24]
    coverage=float(np.isfinite(xy[:,required]).all((1,2)).mean())
    if coverage<.8 or abs(coverage-event['observation_coverage'])>1e-9:
        raise ValueError('Observation coverage mismatch')
    if any(not isinstance(s,str) or len(s)!=64 or any(c not in '0123456789abcdef' for c in s) for s in [state_model_sha256,source_sha256]):
        raise ValueError('Full SHA256 required')
    # Canonical NaN bytes to make input identity independent of NaN payloads.
    xy=xy.copy();xy[np.isnan(xy)]=np.nan
    return dict(schema=VERSION,detector_version=event['version'],protocol=protocol,
                state_model_sha256=state_model_sha256,source_sha256=source_sha256,
                dense_input_sha256=hashlib.sha256(xy.tobytes()).hexdigest(),
                image_dimensions=[width,height],actual_tick_range=[start,end],auto_phase_windows={p:list(windows[p]) for p in PHASES},
                observation=dict(hz=10,border_fraction=.02,minimum_coverage=.8,actual_coverage=coverage),
                evidence_kind='automatic_ordered_attempt_not_reviewed_clip',
                permissions=dict(manual_phase_windows=False,complete_attempt_reviewed=False,
                                 projection_reviewed=False,quality_grade=False,total_score=False),
                validated_for_activation=False)


def existing_engine_options(evidence):
    if evidence.get('schema')!=VERSION or evidence.get('validated_for_activation') is not False or evidence.get('permissions')!={
        'manual_phase_windows':False,'complete_attempt_reviewed':False,
        'projection_reviewed':False,'quality_grade':False,'total_score':False}:
        raise ValueError('Unrecognized or elevated automatic evidence')
    return dict(view='front',complete_attempt=False,phase_windows=None,projection_reviewed=False)
