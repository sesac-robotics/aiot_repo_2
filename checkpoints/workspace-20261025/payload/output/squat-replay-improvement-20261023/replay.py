from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
P=Path(__file__).parent;L=P.parent/'live-webcam-20261023';SESSION=L/'sessions/1792727328056605892';sys.path.insert(0,str(L))
from server import score_samples
from movement import MovementTracker
from runtime_snapshot import digest,read_source
z=np.load(SESSION/'pose.npz');ticks=z['frame_numbers'];xy=z['image_xy'];w=int(z['width']);h=int(z['height']);old=json.loads((SESSION/'final-state.json').read_text())
tracker=MovementTracker('squats','front');segments=[];rows=[];eventinfo=[]
for tick,x in zip(ticks,xy):
 seg=tracker.push(int(tick),x)
 if not seg:continue
 segments.append(seg);score=score_samples(seg,w,h,'squats','front',True)
 row=dict(number=tracker.count,status='저장 관절 재실행 · 검증 중',duration=(seg[-1][0]-seg[0][0])/10,front_candidate=score['front_candidate'],coaching=score['coaching'],accepted_items=score['accepted_items'])
 rows.append(row);eventinfo.append(dict(number=tracker.count,start_seconds=round((seg[0][0]-ticks[0])/10,1),end_seconds=round((seg[-1][0]-ticks[0])/10,1)))
# Evaluate safeguards on the same old boundaries: do not confuse segmentation with score/model changes.
spec=importlib.util.spec_from_file_location('previous_movement',P/'before/movement.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);tr=mod.MovementTracker('squats','front');matched=[]
for tick,x in zip(ticks,xy):
 seg=tr.push(int(tick),x)
 if seg:
  score=score_samples(seg,w,h,'squats','front',True);matched.append(dict(number=tr.count,before=old['repetitions'][tr.count-1]['accepted_items'],after=score['accepted_items']))
state=dict(old,collecting=False,repetitions=rows,last_rep=rows[-1],movement=tracker.status(),live_result=None,coaching=None,elapsed_seconds=float((ticks[-1]-ticks[0])/10),saved_session_folder=str(P.resolve()),status='저장 관절 재실행 완료 · 추가 촬영 없음',visible=0,front_squat_observation=None)
state['replay']=True;(P/'replay-state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2));(L/'replay-state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
proof=dict(source_session=str(SESSION.resolve()),pose_sha256=hashlib.sha256((SESSION/'pose.npz').read_bytes()).hexdigest(),original_state_sha256=hashlib.sha256((SESSION/'state.json').read_bytes()).hexdigest(),team_digest=digest(read_source()),team_unchanged=digest(read_source())=='c850c28b43de189e32b2419147567247d08385c717c8ca90a6bc2c6d1f58d891',models_unchanged=hashlib.sha256((P.parent/'squat-relabel-expanded-20261023/candidate.joblib').read_bytes()).hexdigest()==json.loads((P.parent/'squat-relabel-expanded-20261023/report.json').read_text())['model_sha256'],before_count=old['movement']['count'],after_count=tracker.count,new_events=eventinfo,same_boundary_comparison=matched,independent_visual_ground_truth=False,additional_recording=False)
(P/'comparison.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2));print('count',proof['before_count'],'->',proof['after_count']);print(eventinfo)
for r in rows:print(r['number'],r['front_candidate']['ai_assisted_points'],[(i['id'],i['grade'],i.get('observation_grade'),i.get('reason'),i['coverage']) for i in r['accepted_items']['items']])
# Use a standard plotting library if available, preserving all gaps rather than interpolating.
try:
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 hip=xy[:,[23,24],1].mean(1);fig,ax=plt.subplots(figsize=(13,4));ax.plot((ticks-ticks[0])/10,hip,lw=1,color='#185b69');ax.invert_yaxis();ax.set(xlabel='Seconds since start',ylabel='Observed pelvis y (pixels)',title='Stored pose replay: pelvis descent and return (not video ground truth)')
 for e in eventinfo:ax.axvspan(e['start_seconds'],e['end_seconds'],alpha=.1);ax.text((e['start_seconds']+e['end_seconds'])/2,np.nanmax(hip),str(e['number']),ha='center')
 for e in tracker.uncertain_attempts:ax.axvspan((e['start_tick']-ticks[0])/10,(e['end_tick']-ticks[0])/10,color='orange',alpha=.3)
 fig.tight_layout();fig.savefig(P/'cycles.png',dpi=140)
except ImportError:pass
