from pathlib import Path
import json,hashlib
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/four-exercise-review-train-v1-20261025';N=O/'reconciled-v2';N.mkdir(exist_ok=True);rows=json.loads((O/'score-inputs.frozen.json').read_text());prior=json.loads((B/'output/shoulder-expanded-train-v3-20261025/visual-labels.frozen.json').read_text());prior={"r2d2_"+r['original_annotation_id']:r for r in prior['team_clips']};changes=[]
for r in rows:
 p=prior.get(r['id'])
 if p is None or p['lower_grade'] is not None:continue
 for j,x in enumerate(r['items']):
  if x['item']!='lower_return':continue
  old=r['levels'][j];r['levels'][j]=None;x['eligible']=False;x.setdefault('holds',[]).append('reused_prior_exact_clip_lower_unobservable_or_nonbilateral_protocol');changes.append(dict(id=r['id'],exercise=r['exercise'],item=x['item'],bank=x['bank'],old=old,new=None,eligible=False,reasons=[p['note_ko']],prior_evidence=p['review_pages'],original_clip_window_seconds=p['original_clip_window_seconds']))
for name,v in [('score-inputs.frozen.json',rows),('prior-lower-reconciliation.frozen.json',changes)]:
 p=N/name;assert not p.exists();p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
s=(B/'tools/four-exercise-review/train.py').read_text().replace("O=B/'output/four-exercise-review-train-v1-20261025'","O=B/'output/four-exercise-review-train-v1-20261025/reconciled-v2'");(B/'tools/four-exercise-review/train_reconciled.py').write_text(s)
print('held_lower',len(changes),'unique',len({r['id'] for r in changes}))
