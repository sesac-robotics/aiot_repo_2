"""Align existing visual Train labels to actual event pixels; no grades from geometry."""
from pathlib import Path
import sys,json,hashlib,collections
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-rebuild-v2-20261025';sys.path.insert(0,str(B/'tools/shoulder-work'))
from observation_features_v4 import dense_samples,extract_body
from phase_features_v3 import angle
from anatomical_turn_features_v1 import extract as extract_turn
plan=json.loads((O/'plan.frozen.json').read_text())['rows'];out=[]
for r in plan:
 p=O/'observations'/r['id']/'primary.npz' if r['requires_extraction'] else Path(r['cache']['path']);m=json.loads(p.with_suffix('.json').read_text());z=np.load(p);assert m['config']['source_sha256']==r['source_sha256'];assert str(z['fingerprint'])==m['fingerprint'];assert hashlib.sha256(json.dumps(m['config'],sort_keys=True).encode()).hexdigest()==m['fingerprint']
 native=z['frame_numbers'];times=native/r['fps'];nominal=np.rint((times-times[0])*10).astype(int)+round(times[0]*10);assert len(set(nominal))==len(nominal),'source slots cannot be silently merged'
 mask=(times>=r['context_seconds'][0]-.04)&(times<=r['context_seconds'][1]+.04);ns=native[mask];ss=list(zip(nominal[mask],z['image_xy'][mask]));ts,x=dense_samples(ss,r['width'],r['height']);frames=np.full(len(ts),np.nan);frames[np.array([t for t,_ in ss])-ts[0]]=ns
 record=dict(r,cache_path=str(p),cache_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cache_fingerprint=m['fingerprint'],delegate=m['config']['delegate'],context_native_frames=ns.tolist(),actual_nominal_slots=[int(t) for t,_ in ss],lower_status='held',body_status='held',reason=None)
 a,b=r['reviewed_turn_seconds'];turn=np.where((frames/r['fps']>=a-.01)&(frames/r['fps']<=b+.01))[0]
 if len(turn)<3:record['reason']='fewer_than_three_actual_turn_slots';out.append(record);continue
 # Use middle actual three samples in the manually reviewed turn interval.
 mid=(len(turn)-3)//2;turn=turn[mid:mid+3];el=np.stack([angle(x,23,11,13),angle(x,24,12,14)],1);signal=np.mean(el,1)
 # Peak is source-observed prior motion, never a copied ready baseline or inferred label.
 possible=np.arange(1,max(1,turn[0]-2));possible=possible[np.isfinite(signal[possible-1])&np.isfinite(signal[possible])&np.isfinite(signal[possible+1])]
 if not len(possible):record['reason']='no_observed_preceding_upper_context';out.append(record);continue
 center=int(possible[np.argmax([np.mean(signal[i-1:i+2]) for i in possible])]);peak=np.arange(center-1,center+2)
 if not np.isfinite(frames[peak]).all():record['reason']='peak_native_frames_missing';out.append(record);continue
 event=dict(protocol='shoulder_press',laterality='bilateral',posture=r['posture'],closure='observed_rack',rack_visible_stable=True,peak_frames=[int(frames[peak[0]]),int(frames[peak[-1]])],turn_frames=[int(frames[turn[0]]),int(frames[turn[-1]])])
 f=extract_turn(ss,event,r['fps'],r['width'],r['height'],'front',native_frame_numbers=ns)
 start=int(ts[peak[0]]);end=int(ts[turn[-1]]);body=extract_body(ss,[start,end],True,r['width'],r['height'],'front')
 record.update(event=event,feature_window_seconds=[start/10,end/10],lower_status=f['status'],body_status=body['status'],lower=f,body=body,reason=f.get('reason'),visual_review_status='pending',grade_source='existing_visual_event_label_not_geometry',initial_ready_inferred=False)
 out.append(record)
(O/'event-feature-candidates.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('events',len(out),'lower',collections.Counter(r['lower_status'] for r in out),'body',collections.Counter(r['body_status'] for r in out));print('reasons',collections.Counter(r.get('reason') for r in out));
