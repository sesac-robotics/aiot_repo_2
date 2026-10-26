from pathlib import Path
import json,sys,collections,numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');sys.path.insert(0,str(B/'tools/shoulder-work'));from phase_features_v3 import extract
O=B/'output/shoulder-work-20261024/normal-reference-v2';R=B/'output/shoulder-work-20261024/full-review-v1';dims={r['id']:r for r in json.loads((R/'render-receipts.json').read_text())};rows=[]
for r in json.loads((O/'plan.frozen.json').read_text())['rows']:
 p=O/'observations'/r['id']/'primary.npz';m=json.loads(p.with_suffix('.json').read_text());z=np.load(p);times=z['time_seconds'];a,b=r['return_event'] or r['target_context'];attempts=[]
 for context in [3,2,1.5,4,5]:
  mask=(times>=max(0,a-context))&(times<=b+.1);slots={}
  for t,x in zip(times[mask],z['image_xy'][mask]):
   k=int(round((t-times[0])*10))+int(round(times[0]*10));err=abs(t-k/10)
   if k not in slots or err<slots[k][0]:slots[k]=(err,x)
  f=extract([(k,v[1]) for k,v in sorted(slots.items())],dims[r['id']]['width'],dims[r['id']]['height']);match=f['status']=='features_ready' and a-.14<=np.median(f['return_ticks'])/10<=b+.14
  attempts.append(dict(context_seconds=context,status=f['status'],reason=f.get('reason','matched_visual_return' if match else 'phase_return_time_does_not_match_selected_visual_label'),coverage=f.get('coverage'),return_ticks=f.get('return_ticks'),visual_return_match=bool(match)))
 rows.append(dict(id=r['id'],view=r['view'],split=r['split'],new_cache=str(p),selected_visual_return=r['return_event'],attempts=attempts,any_matching_phase=any(x['visual_return_match'] for x in attempts)))
(O/'source-feature-diagnostics.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n');print(collections.Counter(x['reason'] for r in rows for x in r['attempts']));print('matching',sum(r['any_matching_phase'] for r in rows),'of',len(rows))
