from pathlib import Path
from datetime import datetime,timezone
import json,sys,hashlib
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');L=B/'output/live-webcam-20261023';O=Path(__file__).parent
sys.path.insert(0,str(L));sys.path.insert(0,str(B/'tools/shoulder-work'))
from shoulder_candidate import ShoulderTracker
from phase_features_v3 import extract as previous_return
from score_items_v4 import score
F=L/'sessions/1792849624147364792';z=np.load(F/'pose.npz');raw_sha=hashlib.sha256((F/'pose.npz').read_bytes()).hexdigest();tr=ShoulderTracker();rows=[]
for tick,xy in zip(z['frame_numbers'],z['image_xy']):
    seg=tr.push(int(tick),xy,width=int(z['width']),height=int(z['height']))
    if not seg:continue
    window=[seg[0][0],seg[-1][0]]
    # These are previously detected exercise attempts, not video-adjudicated
    # quality labels. Explicit-window confirmation is a functional probe only.
    native=score(seg,models=None,width=int(z['width']),height=int(z['height']),trunk_window=window,exercise_confirmed=True)
    old=previous_return(seg,int(z['width']),int(z['height']))
    noarms=[]
    for t,p in seg:
        p=p.copy();p[13:17]=np.nan;noarms.append((t,p))
    masked=score(noarms,models=None,width=int(z['width']),height=int(z['height']),trunk_window=window,exercise_confirmed=True)
    unconfirmed=score(seg,models=None,width=int(z['width']),height=int(z['height']),trunk_window=window,exercise_confirmed=False)
    a=native['features']['frontal_trunk_hold'];b=masked['features']['frontal_trunk_hold']
    assert a==b, 'body path must be independent of arm visibility'
    assert unconfirmed['features']['frontal_trunk_hold']['status']=='held'
    assert all(i['points'] is None and i['grade'] is None for i in native['items'])
    def compact(f):return {k:v for k,v in f.items() if k in ['status','reason','version','reference_ticks','return_ticks','coverage','window_ticks','baseline_ticks','baseline_semantics','quality']}
    rows.append(dict(number=tr.count,segment_ticks=window,prior_v3=compact(old),new_return=compact(native['features']['lower_return']),new_body=compact(a),body_same_after_arms_removed=True,unconfirmed_body_reason=unconfirmed['features']['frontal_trunk_hold']['reason'],new_item_scores=native['items']))
assert len(rows)==7;assert hashlib.sha256((F/'pose.npz').read_bytes()).hexdigest()==raw_sha
files=[B/'tools/shoulder-work'/n for n in ['observation_features_v4.py','score_items_v4.py','phase_features_v3.py']]
summary={'created_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'saved seven-attempt integration diagnostics; not training, quality truth or accuracy','session':str(F),'raw_sha256':raw_sha,'completed_replay_attempts':len(rows),'new_return_features_ready':sum(r['new_return']['status']=='features_ready' for r in rows),'new_body_features_ready':sum(r['new_body']['status']=='features_ready' for r in rows),'prior_return_features_ready':sum(r['prior_v3']['status']=='features_ready' for r in rows),'all_arms_removed_body_features_equal':True,'all_unconfirmed_body_windows_held':True,'all_new_AI_item_points_held':True,'feature_files_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'new_models_trained':False,'production_changed':False,'camera_capture_started':False,'rows':rows}
(O/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','feature_files_sha256']},ensure_ascii=False))
