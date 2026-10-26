"""Train-only choice between two calibrated cues and their conservative union."""
import json,sys,datetime
from pathlib import Path
B=Path(__file__).resolve().parents[2];O=B/'output/mild-return-combined-20261024';O.mkdir(exist_ok=True);P=B/'output/mild-return-calibrated-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'))
from train_scores import metrics
from common import sha
protocol=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent=str(P),candidates=['upper','reference','union'],selection='Train group OOF only: require mild coverage 5/5, recall>=.65, normal FP<=.15; then maximize mild recall, total error recall, minimize normal FP and balanced group MAE',development_used_for_selection=False,Test_used_for_selection=False,iterative_development_experiment=True,code_sha256=sha(Path(__file__)))
f=O/'COMBINATION_PROTOCOL.frozen.json';assert not f.exists();f.write_text(json.dumps(protocol,indent=2))
read=lambda p:json.loads(p.read_text());reports=read(P/'changed-heads-fit/score-training.json')['heads'];upper=next(h for h in reports if h['key'][2]=='height_upper_reversal_v1');ref=next(h for h in reports if h['key'][2]=='height_reference_pose_v1');a={p['id']:p for p in upper['selected_oof_predictions']};b={p['id']:p for p in ref['selected_oof_predictions']};assert a.keys()==b.keys()
# Both models used the same participant-held outer folds.
for u,r in zip(upper['nested_CV_folds'],ref['nested_CV_folds']):assert set(u['outer_ids'])==set(r['outer_ids']) and set(u['fit_groups'])==set(r['fit_groups'])
candidates=[];mild=set(upper['mild_Train_ids'])
for name in ['upper','reference','union']:
 ps=[]
 for i,u in a.items():
  r=b[i];assert u['target']==r['target'];v=u['prediction'] if name=='upper' else r['prediction'] if name=='reference' else min(u['prediction'],r['prediction']);ps.append(dict(u,prediction=v))
 m=metrics(ps);mm=metrics([p for p in ps if p['id'] in mild]);rank=[not(m['error_recall']>=.65 and m['normal_false_alarm']<=.15),1-mm['error_recall'],1-m['error_recall'],m['normal_false_alarm'],m['balanced_group_MAE']];candidates.append(dict(name=name,rank=rank,metrics=m,mild_metrics=mm,predictions=ps))
best=min(candidates,key=lambda c:c['rank']);(O/'TRAIN_SELECTION.frozen.json').write_text(json.dumps(dict(selected=best['name'],candidates=candidates,selection_before_development_evaluation=True),indent=2));print([(c['name'],c['rank']) for c in candidates]);print('SELECTED',best['name'])
