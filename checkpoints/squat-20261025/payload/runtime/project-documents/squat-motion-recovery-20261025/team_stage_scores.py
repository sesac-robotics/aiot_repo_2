import sys,json,copy
from pathlib import Path
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');A=B/'output/live-webcam-20261023';O=B/'project-documents/squat-motion-recovery-20261025';sys.path[:0]=[str(O),str(A)]
from movement_stage_candidate import MovementTracker
import importlib.util
spec=importlib.util.spec_from_file_location('checklist_comparison',O/'checklist_comparison_candidate.py');comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison);comparison.ROOT=A;sys.modules['checklist_comparison']=comparison
from server import score_samples,finish_observation_window
profile=json.loads((B/'output/squat-count-Train-profile-20261025/natural-knee-gaps.frozen.json').read_text());rows=[]
for sid in ['V10','V18']:
 src=next(r for r in profile['sources']if r['id']==sid);z=np.load(B/'output/integrated-five-20261024/team-observations'/sid/'side.npz');available=list(zip(np.rint(z['time_seconds']*10).astype(int),z['image_xy']));t=MovementTracker('squats','front')
 for tick,xy in available:
  seg=t.push(int(tick),xy,width=src['width'],height=src['height'])
  if not seg:continue
  ret=finish_observation_window(seg,t.standing_reference,available)
  r=score_samples(seg,src['width'],src['height'],'squats','front',standing_reference=t.standing_reference,return_samples=ret);a=r['score_100']
  rows.append(dict(source=sid,number=t.count,window=[seg[0][0],seg[-1][0]],observation=t.completion_receipts[-1]['observation'],ai=a['ai'],rule=a['rule'],items=[{k:i.get(k)for k in ['id','ai_points','rule_points','reasons']}for i in a['items']]))
(O/'team-stage-scores.json').write_text(json.dumps({'scope':'Train exposed engineering replay, same frozen scoring models, all actual raw NaN values preserved; scoring eligibility separate from counting','rows':rows},ensure_ascii=False,indent=2)+'\n');print([(r['source'],r['window'],r['ai']['assessed_weight'],r['observation']['knee_phase_evidence_coverage'])for r in rows])
