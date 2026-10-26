"""Post-freeze diagnostic comparison; cannot feed model selection."""
import json,collections
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');OLD=B/'output/label-verified-five-expanded-20261024';O=B/'output/pushup-repaired-five-v2-20261024'
def read(p):return json.loads(Path(p).read_text())
a=read(OLD/'test-evaluation.json');b=read(O/'test-evaluation.json');old={(r['id'],r['criterion']):r for r in a['predictions']};new={(r['id'],r['criterion']):r for r in b['predictions']};common=old.keys()&new.keys();assert all(old[k]['target']==new[k]['target'] for k in common)
def count(rows):
 return dict(scored=len(rows),normal=sum(r['target']==2 for r in rows),false_alarms=sum(r['target']==2 and r['prediction']<1.5 for r in rows),errors=sum(r['target']<2 for r in rows),detected_errors=sum(r['target']<2 and r['prediction']<1.5 for r in rows))
criteria=sorted({r['criterion'] for r in a['predictions']+b['predictions'] if r['exercise']=='pushups'});out=dict(diagnostic_not_independent_final_test=True,selection_used_test_results=False,original_test_labels_unchanged=True,baseline=str(OLD),candidate=str(O),pushups_old=count([r for r in old.values() if r['exercise']=='pushups']),pushups_new=count([r for r in new.values() if r['exercise']=='pushups']),matched_old=count([old[k] for k in common if old[k]['exercise']=='pushups']),matched_new=count([new[k] for k in common if new[k]['exercise']=='pushups']),by_criterion={c:dict(old=count([r for r in old.values() if r['exercise']=='pushups' and r['criterion']==c]),new=count([r for r in new.values() if r['exercise']=='pushups' and r['criterion']==c])) for c in criteria},became_unscored=[list(k) for k in old.keys()-new.keys()],newly_scored=[list(k) for k in new.keys()-old.keys()],other_four_predictions_unchanged=all(abs(old[k]['prediction']-new[k]['prediction'])<1e-10 for k in common if old[k]['exercise']!='pushups'))
(O/'diagnostic-comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
