"""Queue the next visual review using training OOF errors only, never Test results."""
import json,collections
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/pushup-repaired-five-v2-20261024';reports=json.loads((O/'score-training.json').read_text())['heads'];rows={r['id']:r for r in json.loads((O/'score-inputs.frozen.json').read_text())};queue=[];support=[]
for h in reports:
 support.append(dict(key=h['key'],status=h['status'],normal_groups=h.get('normal_groups'),error_groups=h.get('error_groups'),development_gate=h.get('candidate_gate_passed',False)))
 for p in h.get('selected_oof_predictions',[]):
  r=rows[p['id']];assert r['split']=='train'
  wrong=(p['target']==2 and p['prediction']<1.5) or (p['target']<2 and p['prediction']>=1.5)
  if not wrong:continue
  queue.append(dict(id=p['id'],person=p['person'],group=p['group'],key=h['key'],target=p['target'],training_oof_prediction=p['prediction'],absolute_error=abs(p['target']-p['prediction']),sources=r['sources'],split='train',label_change_requires_visual_evidence=True))
queue.sort(key=lambda r:(-r['absolute_error'],r['id'],r['key']));out=[];seen=collections.Counter()
for r in queue:
 k=(r['key'][0],r['key'][1],r['person'])
 if seen[k]>=2:continue
 seen[k]+=1;out.append(r)
(O/'TRAIN_VISUAL_REVIEW_QUEUE.json').write_text(json.dumps(dict(selection='Training out-of-fold errors only; Test predictions not read.',items=out),indent=2));(O/'criterion-support.json').write_text(json.dumps(support,indent=2));print('TRAIN_ONLY_REVIEW_QUEUE',len(out),'unsupported_banks',sum(r['status']!='trained_experimental' for r in support))
