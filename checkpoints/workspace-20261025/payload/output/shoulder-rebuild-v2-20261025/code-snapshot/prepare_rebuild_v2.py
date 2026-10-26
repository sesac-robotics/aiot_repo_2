from pathlib import Path
import json,collections,hashlib,sys
import numpy as np,cv2
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-rebuild-v2-20261025';R=B/'output/shoulder-work-20261024/full-review-v1';sys.path[:0]=[str(B/'tools/error-enrichment')];from split_guard import assert_training_allowed
labels=[r for r in json.loads((R/'combined-selected-labels.frozen.json').read_text()) if r['split']=='train' and r['view']=='front'];inv={r['id']:r for r in json.loads((R/'inventory.frozen.json').read_text())['sources']};wanted={r['source_sha256'] for r in labels};cache=collections.defaultdict(list)
for p in (B/'output').rglob('primary.json'):
 if any(v in p.parts for v in ['snapshots','before','code-snapshot']):continue
 try:
  d=json.loads(p.read_text());c=d.get('config',{})
  if c.get('source_sha256') not in wanted or c.get('sample_hz')!=10 or not p.with_suffix('.npz').exists():continue
  z=np.load(p.with_suffix('.npz'));fs=z['frame_numbers'];fps=c['segment']['fps']
  if str(z['fingerprint'])!=d['fingerprint']:continue
  cache[c['source_sha256']].append(dict(path=str(p.with_suffix('.npz')),fingerprint=d['fingerprint'],start=float(fs[0]/fps),end=float(fs[-1]/fps),fps=fps,frames=len(fs)))
 except (ValueError,OSError,KeyError):pass
plan=[];excluded=[]
for l in labels:
 s=inv[l['id']];assert s['split']=='train';assert_training_allowed([dict(s,sources=[s['source_sha256']])])
 if s['person'] in ['shoulder_005','shoulder_008','shoulder_009']:excluded.append(dict(id=s['id'],reason='actual_rear_view_separate_bank_pending'));continue
 cap=cv2.VideoCapture(s['path']);fps=cap.get(5);last=(cap.get(7)-1)/fps;w=int(cap.get(3));h=int(cap.get(4));cap.release()
 for n,(a,b) in enumerate(l['observed_events']):
  # Labels identify the actual turn; preceding 2s is input context, never a ready baseline.
  if b-a<.19:excluded.append(dict(id=s['id'],event=[a,b],reason='reviewed_turn_too_short_for_three_actual_samples'));continue
  lo=max(0,a-2);hi=min(last,b+.4)
  opts=[c for c in cache[s['source_sha256']] if c['start']<=lo+.04 and c['end']>=b-.04]
  # Fixed source-context coverage preference, independent of grades and feature values.
  opts.sort(key=lambda c:(-(min(c['end'],hi)-max(c['start'],lo)),-c['frames'],c['path']))
  pick=opts[0] if opts else None
  plan.append(dict(id=s['id']+'_event_'+str(n+1),source_id=s['id'],path=s['path'],source_sha256=s['source_sha256'],sources=[s['source_sha256']],person=s['person'],recording_group=s['recording_group'],split='train',posture='seated' if s['person'] in ['shoulder_011','shoulder_023'] else 'standing',view='front',width=w,height=h,fps=fps,reviewed_turn_seconds=[a,b],context_seconds=[lo,hi],lower_grade=l['lower_return_grade'],body_grade=l['frontal_trunk_grade'],label_source=str(R/'combined-selected-labels.frozen.json'),reviewed_source_windows=l['reviewed_windows'],cache=pick,requires_extraction=pick is None))
assert_training_allowed(plan)
obj={'schema':'shoulder-manually-aligned-Train-rebuild-v2','selection':'all existing Train/front selected observed return events; rear views excluded; short events held; no Test/validation transfer','rows':plan,'excluded':excluded,'training_source_roles_changed':False,'label_hash':hashlib.sha256((R/'combined-selected-labels.frozen.json').read_bytes()).hexdigest()}
p=O/'plan.frozen.json';assert not p.exists();p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');print('events',len(plan),'cached',sum(not r['requires_extraction'] for r in plan),'missing',sum(r['requires_extraction'] for r in plan),'excluded',len(excluded));print('missing events',[(r['id'],r['context_seconds']) for r in plan if r['requires_extraction']])
