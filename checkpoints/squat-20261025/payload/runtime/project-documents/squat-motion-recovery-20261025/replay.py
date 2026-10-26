import sys,importlib.util,json,copy,hashlib,datetime
from pathlib import Path
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'project-documents/squat-motion-recovery-20261025';A=B/'output/live-webcam-20261023';sys.path.insert(0,str(A))
def load(n,p):
 spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=load('old_movement',O/'movement.before.py');new=load('candidate_movement',O/'movement_candidate.py')
from shoulder_candidate import ShoulderTracker
from server import score_samples,finish_observation_window
all_results=[]
for f in sorted((A/'sessions').glob('*/final-state.json'),key=lambda p:int(p.parent.name)):
 if not (f.parent/'pose.npz').exists():continue
 state=json.loads(f.read_text());z=np.load(f.parent/'pose.npz');w=int(z['width']);h=int(z['height']);available=[(int(t),p)for t,p in zip(z['frame_numbers'],z['image_xy'])];exercise=state['exercise'];view=state['view'];versions={}
 for version,module in [('before',old),('candidate',new)]:
  tracker=ShoulderTracker(exercise,view)if exercise=='shoulder'else module.MovementTracker(exercise,view);segments=[]
  for tick,pts in available:
   seg=tracker.push(tick,pts,width=w,height=h)
   if seg:
    ref=copy.deepcopy(tracker.standing_reference);score=None
    # Identical windows share cached scoring; only changed inputs are rescored.
    signature=(exercise,view,tuple(t for t,_ in seg))
    cache=globals().setdefault('cache',{})
    if signature in cache:score=cache[signature]
    else:
     try:
      returns=finish_observation_window(seg,ref,available)if exercise=='squats'and view=='front'and ref is not None else None
      result=score_samples(seg,w,h,exercise,view,standing_reference=ref,return_samples=returns)
      a=result['score_100'];score={k:a.get(k)for k in ['ai','rule']};score['items']=[{k:i.get(k)for k in ['id','ai_points','rule_points']}for i in a['items']]
     except Exception as e:score={'held':str(e)}
     cache[signature]=score
    segments.append(dict(start_tick=seg[0][0],end_tick=seg[-1][0],samples=len(seg),score=score))
  versions[version]={'count':tracker.count,'segments':segments,'status':tracker.status()}
 r={'session':f.parent.name,'exercise':exercise,'view':view,'samples':len(available),'saved_count':state['movement']['count'],**versions};r['counts_equal']=r['before']['count']==r['candidate']['count'];r['segments_and_scores_equal']=r['before']['segments']==r['candidate']['segments'];all_results.append(r)
 print(f.parent.name,exercise,'count',r['before']['count'],r['candidate']['count'],'same',r['segments_and_scores_equal'],flush=True)
verification={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sessions':len(all_results),'counts_all_equal':all(r['counts_equal']for r in all_results),'changed_sessions':[r['session']for r in all_results if not r['segments_and_scores_equal']],'results':all_results,'scope':'Same original saved raw poses and existing scoring models; not accuracy ground truth. Each version uses its actual segment and at most 3 observed post slots for return.'}
(O/'replay.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
