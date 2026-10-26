from pathlib import Path
from datetime import datetime,timezone
import json,sys
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');O=Path(__file__).parent
sys.path.insert(0,str(B/'tools/shoulder-work'));sys.path.insert(0,str(B/'output/live-webcam-20261023'))
from return_context_v1 import ReturnContextQueue
from shoulder_candidate import ShoulderTracker
from score_items_v4 import score
z=np.load(B/'output/live-webcam-20261023/sessions/1792849624147364792/pose.npz')
def run(cutoff=None):
    q=ReturnContextQueue();tr=ShoulderTracker();results=[]
    def collect(event,arrival):
        assert event['evaluated_at_tick']<=arrival
        assert all(t<=arrival for t,p in event['feature_input_samples'])
        assert max(t for t,p in event['feature_input_samples'])<=event['original_window'][1]+3
        event=dict(event);event['emitted_at_tick']=arrival;results.append(event)
    last=None
    for tick,x in zip(z['frame_numbers'],z['image_xy']):
        tick=int(tick)
        if cutoff is not None and tick>cutoff:break
        last=tick
        for event in q.push(tick,x):collect(event,tick)
        segment=tr.push(tick,x,width=int(z['width']),height=int(z['height']))
        if segment:
            event=q.submit(tr.count,segment,int(z['width']),int(z['height']))
            if event:collect(event,tick)
    for event in q.finish():collect(event,last)
    return tr,results
tr,events=run();assert tr.count==len(events)==7
rows=[]
for event in events:
    assert event['features']['status']=='features_ready'
    out=score(event['feature_input_samples'],models=None,trunk_window=event['original_window'],exercise_confirmed=True)
    assert out['features']['frontal_trunk_hold']['window_ticks']==event['original_window']
    assert all(i['points'] is None for i in out['items'])
    rows.append(dict(number=event['attempt_id'],original_window=event['original_window'],return_ticks=event['features']['return_ticks'],emitted_at_tick=event['emitted_at_tick'],last_evaluated_input_tick=event['evaluated_at_tick'],added_actual_ticks=event['added_actual_ticks'],delay_slots=event['emitted_at_tick']-event['original_window'][1],all_scores_held=True))
first_end=rows[0]['original_window'][1];cut_tr,cut_events=run(cutoff=first_end)
assert cut_tr.count==1 and len(cut_events)==1
assert cut_events[0]['features']['status']=='held'
assert cut_events[0]['context_reason']=='recording_ended_before_return_context_closed'
assert cut_events[0]['added_actual_ticks']==[]
report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),scope='same exposed saved seven-attempt causal replay and bounded-end functional test',replayed_count=tr.count,causal_features_ready=len(rows),rows=rows,cutoff_at_first_completion_held_without_future_frames=True,production_changed=False,new_model_fit=False,camera_capture=False)
(O/'queue-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
