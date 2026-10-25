from pathlib import Path
import json,sys,hashlib,collections
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');L=B/'output/live-webcam-20261023';O=B/'output/shoulder-work-20261024/phase-model-v1';O.mkdir(exist_ok=True)
sys.path.insert(0,str(L));from shoulder_candidate import ShoulderTracker
sys.path.insert(0,str(B/'tools/shoulder-work'));from phase_features_v1 import extract
sys.path.insert(0,str(B/'tools/error-enrichment'));from split_guard import assert_training_allowed,POLICY
run=Path(json.loads((B/'output/CURRENT_FIVE_EXERCISE_RUN.json').read_text())['run']);rows=json.loads((run/'score-inputs.frozen.json').read_text());assert_training_allowed([r for r in rows if r['split']=='train'])
# Read role/source hashes only from exposed diagnostic test; no fit or selection.
test=json.loads((run/'test-inputs.frozen.json').read_text());assert not {s for r in rows if r['split']=='train' for s in r['sources']} & {s for r in test for s in r['sources']}
support={}
for name in ['lower_return','frontal_trunk_hold']:
 support[name]=[{'split':s,'bank':b,'grade':g,'count':n} for (s,b,g),n in collections.Counter((r['split'],i['bank'],g) for r in rows if r['exercise']=='shoulder' for i,g in zip(r['items'],r['levels']) if i['item']==name and g is not None and i['eligible']).items()]
S=L/'sessions/1792849624147364792';d=json.loads((S/'final-state.json').read_text());p=np.load(S/'pose.npz');t=ShoulderTracker();records=[]
for tick,xy in zip(p['frame_numbers'],p['image_xy']):
 seg=t.push(int(tick),xy,int(p['width']),int(p['height']))
 if not seg:continue
 r=d['repetitions'][len(records)];f=extract(seg,int(p['width']),int(p['height']))
 a=np.asarray([xy for _,xy in seg]);from phase_features_v1 import angle
 el=np.stack([angle(a,11,13,15),angle(a,12,14,16)],1).max(1);ev=np.stack([angle(a,23,11,13),angle(a,24,12,14)],1).max(1);signal=np.maximum(el,ev);n=min(5,max(2,round(len(a)*.15)))
 records.append({'number':r['number'],'features':f,'legacy_window_ticks':[int(x) for x,_ in seg[-n:]],'new_return_ticks':f.get('return_ticks'),'legacy_measurement':next(i['measured'] for i in r['items'] if i['id']=='lower_return'),'replicated_legacy_median':float(np.median(signal[-n:])),'actual_three_return_median':float(np.median(signal[-3:])),'reference_elbow_elevation':[float(np.median(el[:6])),float(np.median(ev[:6]))],'inferred_quality_label':None})
report={'feature_schema':'shoulder-ready-return-relative-v1','fixed_current_run':str(run),'support':support,'live_records':records,'live_feature_ready':sum(r['features']['status']=='features_ready' for r in records),'existing_roles_preserved':True,'training_performed':False,'grade_prediction_enabled':False,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [POLICY,run/'score-inputs.frozen.json',run/'test-inputs.frozen.json',B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json',S/'pose.npz',B/'tools/shoulder-work/phase_features_v1.py']}}
(O/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'support':support,'live_feature_ready':report['live_feature_ready'],'windows':[{k:r[k] for k in ['number','legacy_measurement','replicated_legacy_median','actual_three_return_median']} for r in records]},ensure_ascii=False))
