import sys,json,importlib.util
from pathlib import Path
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');A=B/'output/live-webcam-20261023';O=B/'project-documents/squat-motion-recovery-20261025';sys.path[:0]=[str(O),str(A)]
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=load('old_tracker_score',O/'movement.before.py');new=load('new_tracker_score',O/'movement_stage_candidate.py');oldc=load('old_compare_score',O/'checklist_comparison.before.py');newc=load('new_compare_score',O/'checklist_comparison_candidate.py');oldc.ROOT=newc.ROOT=A
from server import score_samples,finish_observation_window
profile=json.loads((B/'output/squat-count-Train-profile-20261025/natural-knee-gaps.frozen.json').read_text());rows=[]
for sid in ['V10','V18']:
 src=next(r for r in profile['sources']if r['id']==sid);z=np.load(B/'output/integrated-five-20261024/team-observations'/sid/'side.npz');available=[(int(t),p)for t,p in zip(np.rint(z['time_seconds']*10).astype(int),z['image_xy'])];versions={}
 for name,trackerclass,comparison in [('before',old.MovementTracker,oldc),('candidate',new.MovementTracker,newc)]:
  sys.modules['checklist_comparison']=comparison;t=trackerclass('squats','front');segs=[]
  for tick,xy in available:
   seg=t.push(tick,xy,width=src['width'],height=src['height'])
   if seg:
    ret=finish_observation_window(seg,t.standing_reference,available)
    try:
     r=score_samples(seg,src['width'],src['height'],'squats','front',standing_reference=t.standing_reference,return_samples=ret);a=r['score_100'];score={'ai':a['ai'],'rule':a['rule'],'items':[{k:i.get(k)for k in ['id','ai_points','rule_points','reasons']}for i in a['items']]}
    except Exception as e:score={'held':str(e)}
    segs.append(dict(window=[seg[0][0],seg[-1][0]],score=score))
  versions[name]=segs
 rows.append(dict(source=sid,**versions))
(O/'team-score-comparison.json').write_text(json.dumps({'scope':'Same actual Train pixels, no new fit/labels; full pipeline with item-specific actual observation gates','rows':rows},ensure_ascii=False,indent=2)+'\n');print([(r['source'],[(s['window'],s['score'].get('ai',{}).get('earned_subtotal'))for s in r['before']],[(s['window'],s['score'].get('ai',{}).get('earned_subtotal'))for s in r['candidate']])for r in rows])
