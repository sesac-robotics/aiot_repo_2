from pathlib import Path
import sys,json,hashlib,subprocess,os,datetime
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');sys.path.insert(0,str(B/'output/live-webcam-20261023'))
from shoulder_candidate import ShoulderTracker
from return_context_v1 import ReturnContextQueue
from observation_features_v4 import extract_return
from score_items_v4 import score
O=B/'output/shoulder-work-20261024/observation-return-v4-20261025/saved-context-v1';O.mkdir(exist_ok=True)
p=B/'output/live-webcam-20261023/sessions/1792849624147364792/pose.npz';raw_hash=hashlib.sha256(p.read_bytes()).hexdigest();z=np.load(p)
w,h=int(z['width']),int(z['height']);tr=ShoulderTracker();q=ReturnContextQueue();results=[];originals=[]
for t,xy in zip(z['frame_numbers'],z['image_xy']):
    t=int(t);results.extend(q.push(t,xy))
    seg=tr.push(t,xy,width=w,height=h)
    if seg:
        f=extract_return(seg,w,h);originals.append(dict(number=tr.count,window=[seg[0][0],seg[-1][0]],status=f['status'],reason=f.get('reason')))
        immediate=q.submit(tr.count,seg,w,h)
        if immediate is not None:results.append(immediate)
results.extend(q.finish());assert tr.count==len(results)==7
rows=[]
for r in results:
    f=r['features'];assert f['status']=='features_ready'
    assert max(r['added_actual_ticks'])<=r['original_window'][1]+3
    assert f['top_tick']<=r['original_window'][1]
    scored=score(r['feature_input_samples'],width=w,height=h,trunk_window=r['original_window'],exercise_confirmed=True)
    assert all(x['points'] is None and x['grade'] is None for x in scored['items'])
    rows.append(dict(attempt=r['attempt_id'],original_window=r['original_window'],
        ready_reference_ticks=f['reference_ticks'],lift_confirm_tick=f['lift_confirm_tick'],peak_tick=f['top_tick'],
        return_observed_window=f['return_ticks'],return_window_center_tick=float(np.median(f['return_ticks'])),
        confirmed_at_tick=r['evaluated_at_tick'],added_actual_ticks=r['added_actual_ticks'],
        confirmation_delay_seconds=(r['evaluated_at_tick']-r['original_window'][1])/10,
        context_reason=r['context_reason'],body_window=scored['features']['frontal_trunk_hold']['window_ticks'],
        item_scores=scored['items']))
assert hashlib.sha256(p.read_bytes()).hexdigest()==raw_hash
(O/'causal-replay.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
tests=subprocess.run([str(B/'.posture-score-verify/bin/python'),'-m','unittest','discover','-s','tools/shoulder-work','-p','test_return_context_v1.py','-v'],cwd=B,env=env,capture_output=True,text=True)
assert tests.returncode==0; (O/'tests.log').write_text(tests.stdout+tests.stderr)
verification=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='exposed saved-session engineering diagnostics, no video ground truth',
    session_sha256=raw_hash,functional_tests_passed=9,original_system_attempts=7,original_return_features_ready=0,
    causal_context_return_features_ready=7,delay_seconds=[r['confirmation_delay_seconds'] for r in rows],
    event_and_confirmation_times_separate=True,body_window_unchanged=True,
    new_quality_grade_labels=0,new_model_training=False,scored_items=0,production_changed=False,camera_capture=False,
    preserved_originals=originals,
    code_sha256={name:hashlib.sha256((B/'tools/shoulder-work'/name).read_bytes()).hexdigest() for name in ['return_context_v1.py','test_return_context_v1.py','replay_saved_context_v1.py','diagnose_saved_context_v4.py']})
(O/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
code=O/'code-snapshot';code.mkdir(exist_ok=True)
for name in verification['code_sha256']:
    target=code/name;raw=(B/'tools/shoulder-work'/name).read_bytes()
    if target.exists():assert target.read_bytes()==raw
    else:target.write_bytes(raw)
print(json.dumps({k:v for k,v in verification.items() if k not in ['preserved_originals','code_sha256']},ensure_ascii=False))
print([(r['attempt'],r['return_observed_window'],r['confirmed_at_tick']) for r in rows])
