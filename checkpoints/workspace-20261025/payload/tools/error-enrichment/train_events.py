from pathlib import Path
import sys,json,collections,numpy as np,joblib
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/error-enrichment-20261024';sys.path.insert(0,str(B/'five-exercise-comparison'));import upper_support_event_model as event
from plank_partial_item_trial import load_rows
from split_guard import assert_training_allowed
r=[]
for row in load_rows(True):
 if row['split']!='train':continue
 target=0 if row['grades'][3]==2 else 1 if row['id'] in ['error_review_colossus_forward_core','support_pannell_stop'] else None
 if target is None:continue
 x,note=event.event_features(row['seq'])
 if x is not None:r.append(dict(id=row['id'],person=row['person'],group=row['person'],split='train',x=x.tolist(),target=target,source_sha256=row['source']))
base=B/'output/five-exercise-comparison-20261021';direct=json.loads((base/'qevd_upper_event_direct.json').read_text())['reviews'];paths={r['id']:r['extraction']['cache'] for r in json.loads((base/'qevd_upper_event_extraction.json').read_text())['records']}
for d in direct:
 z=np.load(paths[d['id']]);x,note=event.event_features(z['image_xy'])
 if x is not None:r.append(dict(id=d['id'],person=d['person'],group=d['person'],split='train',x=x.tolist(),target=d['visible_upper_support_change'],source_sha256=d['video_sha256']))
r += [v for v in json.loads((O/'upper-events.frozen.json').read_text()) if v['x'] is not None];tr=[v for v in r if v['split']=='train'];va=[v for v in r if v['split']=='validation'];assert not {v['person'] for v in tr}&{v['person'] for v in va};ps=[]
assert_training_allowed(tr)
for person in sorted({v['person'] for v in tr}):
 v=[s for s in tr if s['person']==person];hashes={s['source_sha256'] for s in v};t=[s for s in tr if s['person']!=person and s['source_sha256'] not in hashes];m=event.fit(t);p=m.predict_proba([s['x'] for s in v])[:,list(m.classes_).index(1)];ps += [dict(id=s['id'],person=s['person'],target=s['target'],probability=float(a)) for s,a in zip(v,p)]
def metric(ps):
 return dict(n=len(ps),TP=sum(v['target']==1 and v['probability']>=.75 for v in ps),FN=sum(v['target']==1 and v['probability']<.75 for v in ps),FP=sum(v['target']==0 and v['probability']>=.75 for v in ps),TN=sum(v['target']==0 and v['probability']<.75 for v in ps))
m=event.fit(tr);joblib.dump(dict(model=m,threshold=.75,train_ids=[v['id'] for v in tr],output='visible_upper_support_change_only',whole_body_score=False,production_replacement=False),O/'upper-event-candidate.joblib');vp=[]
if va:
 p=m.predict_proba([v['x'] for v in va])[:,list(m.classes_).index(1)];vp=[dict(id=s['id'],person=s['person'],target=s['target'],probability=float(a)) for s,a in zip(va,p)]
(O/'upper-event-inputs.frozen.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));(O/'upper-event-training.json').write_text(json.dumps(dict(training_n=len(tr),training_people=len({v['person'] for v in tr}),group_cv=metric(ps),development_validation=metric(vp),predictions=ps,validation_predictions=vp,absence_is_not_normal_whole_body_grade=True,threshold_preserved=.75,final_test_used=False,production_replacement=False),ensure_ascii=False,indent=2));print('EVENTS',len(tr),metric(ps),metric(vp))
