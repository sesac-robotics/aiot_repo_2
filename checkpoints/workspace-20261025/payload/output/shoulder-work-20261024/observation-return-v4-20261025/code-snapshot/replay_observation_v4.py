from pathlib import Path
import json,hashlib,collections,sys
import numpy as np
from observation_features_v4 import extract_return,extract_body
from phase_features_v2 import extract as old_return
from phase_features_v3 import extract as v3_return
B=Path('/home/jayden/SeSACPAI-Mid');ROOT=B/'output/shoulder-work-20261024'
O=ROOT/'observation-return-v4-20261025';O.mkdir(exist_ok=True)
R=ROOT/'full-review-v1';V=ROOT/'normal-reference-v2'
inv={r['id']:r for r in json.loads((R/'inventory.frozen.json').read_text())['sources']}
dims={r['id']:r for r in json.loads((R/'render-receipts.json').read_text())}
rows=json.loads((V/'inputs.frozen.json').read_text());bad=json.loads((V/'visual-chain-decisions.json').read_text())['quarantine']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def samples(p,a,b):
    z=np.load(p);m=json.loads(p.with_suffix('.json').read_text())
    assert str(z['fingerprint'])==m['fingerprint']==hashlib.sha256(json.dumps(m['config'],sort_keys=True).encode()).hexdigest()
    times=z['time_seconds'];slots={}
    for t,x in zip(times,z['image_xy']):
        if a-1e-6<=t<=b+1e-6:
            k=int(round((t-times[0])*10))+int(round(times[0]*10))
            assert k not in slots,'do not silently deduplicate original slots'
            slots[k]=x
    return list(sorted(slots.items())),m
out=[]
for r in rows:
    s=inv[r['source_id']];assert s['split']==r['split'] and s['source_sha256']==r['source_sha256']
    assert r['split'] in ('train','validation')
    p=Path(r['cache']);tr=r.get('observed_time_range') or [min(r['phase']['reference_ticks'])/10,max(r['phase']['return_ticks'])/10]
    # Match cached slot phase, not rounded native timestamps, which can be +/- 0.033s.
    ss,m=samples(p,tr[0]-.04,tr[1]+.04)
    assert m['config']['source_sha256']==s['source_sha256']
    kw={'width':dims[s['id']]['width'],'height':dims[s['id']]['height']}
    if r['item']=='frontal_trunk_hold_window_reference':
        window=[ss[0][0],ss[-1][0]]
        f=extract_body(ss,window,True,**kw)
    elif r['item']=='frontal_trunk_hold':
        f=extract_body(ss,[r['phase']['reference_ticks'][0],r['phase']['return_ticks'][-1]],True,**kw)
    else:f=extract_return(ss,**kw)
    out.append(dict(id=r['id'],source_id=s['id'],split=s['split'],previous_visual_quarantine=r['id'] in bad,
                    eligible_for_future_feature_fit=f['status']=='features_ready' and r['id'] not in bad,
                    prior_visual_reason=bad.get(r['id']),status=f['status'],reason=f.get('reason'),features=f,
                    cache=str(p),cache_sha256=sha(p),cache_fingerprint=m['fingerprint'],source_sha256=s['source_sha256']))
# Replay already visually reviewed real early-turnaround failures, without Test inputs.
reg=[]
for sid in ['subject_013_shoulder_bad_front','subject_016_shoulder_good_front']:
    s=inv[sid];assert s['split']=='train'
    p=V/'observations'/sid/'primary.npz'
    ss,m=samples(p,0,3 if '013' in sid else 3.2)
    kw={'width':dims[sid]['width'],'height':dims[sid]['height']}
    reg.append(dict(source_id=sid,split=s['split'],old=old_return(ss,**kw),v3=v3_return(ss,**kw),new=extract_return(ss,**kw),cache_sha256=sha(p)))
manifest={'selection':'all previously frozen v2 reference input rows, prior visual QA labels; not independently held out',
          'rows':len(out),'source_ids':sorted({r['source_id'] for r in rows}),
          'artifacts':{str(p.relative_to(B)):sha(p) for p in [V/'inputs.frozen.json',V/'visual-chain-decisions.json',R/'inventory.frozen.json']},
          'thresholds':{'minimum_projected_width_over_axis':.12,'maximum_width_ratio':4},
          'no_test_sources':True,'new_training':False,'camera_capture':False,'production_changed':False}
manifest_text=json.dumps(manifest,ensure_ascii=False,indent=2)+'\n'
manifest_path=O/'evaluation-manifest.frozen.json'
if manifest_path.exists():assert manifest_path.read_text()==manifest_text,'frozen manifest cannot change'
else:manifest_path.write_text(manifest_text)
(O/'replay.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
(O/'boundary-regression.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2)+'\n')
summary=dict(rows=len(out),roles=dict(collections.Counter(r['split'] for r in out)),
    prior_quarantined=len([r for r in out if r['previous_visual_quarantine']]),
    prior_quarantined_now_held=sum(r['previous_visual_quarantine'] and r['status']=='held' for r in out),
    prior_not_quarantined=len([r for r in out if not r['previous_visual_quarantine']]),
    prior_not_quarantined_now_held=sum(not r['previous_visual_quarantine'] and r['status']=='held' for r in out),
    outcomes=dict(collections.Counter(r['reason'] or r['status'] for r in out)),
    independent_accuracy=False,grade_accuracy_evaluated=False)
(O/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False))
for r in out:
    if r['status']=='held' or r['previous_visual_quarantine']:print(r['id'],r['previous_visual_quarantine'],r['status'],r['reason'])
for r in reg:print(r['source_id'],[(k,r[k].get('return_ticks'),r[k].get('reason')) for k in ['old','v3','new']])
