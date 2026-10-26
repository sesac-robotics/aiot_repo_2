exec(open('/home/jayden/SeSACPAI-Mid/project-documents/squat-motion-recovery-20261025/train_replay.py').read().split("old=load(")[0])
old=load('old_team',O/'movement.before.py');new=load('new_team',O/'movement_candidate.py');profile=json.loads((B/'output/squat-count-Train-profile-20261025/natural-knee-gaps.frozen.json').read_text());labels=json.loads((B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json').read_text())['labels'];rows=[]
for src in profile['sources']:
 sid=src['id'];p=B/'output/integrated-five-20261024/team-observations'/sid/'side.npz';assert hashlib.sha256(p.read_bytes()).hexdigest()==src['cache_sha256'];z=np.load(p);ticks=np.rint(z['time_seconds']*10).astype(int);w=src['width'];h=src['height'];views={r['view']for r in labels if r['source_id']==sid and r['exercise']=='squats'};assert len(views)==1;view=next(iter(views));versions={}
 for version,module in [('before',old),('candidate',new)]:
  t=module.MovementTracker('squats',view);segs=[]
  for tick,xy in zip(ticks,z['image_xy']):
   seg=t.push(int(tick),xy,width=w,height=h)
   if seg:segs.append(dict(start=seg[0][0],end=seg[-1][0],samples=len(seg),observed_fraction=t.last_completion_coverage))
  versions[version]=dict(count=t.count,segments=segs,uncertain=t.uncertain_attempts,recoveries=t.recovery_receipts if hasattr(t,'recovery_receipts')else [])
 rows.append(dict(source=sid,split='train',view=view,source_sha256=src['source_sha256'],width=w,height=h,cache_filename_is_not_view_authority=True,**versions));print(sid,view,versions['before']['count'],versions['candidate']['count'],versions['candidate']['recoveries'],flush=True)
(O/'team-train-replay.json').write_text(json.dumps({'scope':'Actual Train cached raw observations with existing reviewed view and rotated pixel dimensions. Counts are engineering replay, not independent truth. side.npz filename is not selected view.','source_count':len(rows),'rows':rows},ensure_ascii=False,indent=2)+'\n')
