from pathlib import Path
import json,joblib,numpy as np
from standing_state import make_reference,evaluate_return
O=Path(__file__).resolve().parent;S=O.parent/'squat-return-20261023';labels=json.loads((S/'labels.frozen.json').read_text())['labels'];b=joblib.load(O/'model.joblib');results=[]
for r in labels:
 name=Path(r['path']).stem;refpath=O/f'{name}-reference.npz'
 if r['grade'] is None or not refpath.exists():continue
 z=np.load(S/f'{name}-heavy.npz');ref=make_reference(np.load(refpath)['image_xy']);mask=(z['frame_numbers']>=r['first_frame'])&(z['frame_numbers']<=r['last_frame']+6);p=evaluate_return(z['image_xy'][mask],ref,int(z['width']),int(z['height']))
 results.append(dict(id=r['id'],person=r['person'],split=r['split'],target=r['grade'],prediction=p))
held=[r for r in results if r['split']=='validation'];m=dict(n=len(held),assessed=sum(r['prediction'].get('ai_grade') is not None for r in held),partial_n=sum(r['target']==1 for r in held),partial_confirmed_standing=sum(r['target']==1 and r['prediction'].get('standing_confirmed',False) for r in held),normal_n=sum(r['target']==2 for r in held),normal_unconfirmed=sum(r['target']==2 and not r['prediction'].get('standing_confirmed',False) for r in held));(O/'return-validation.json').write_text(json.dumps(dict(fixed_validation=m,results=results,training_predictions_are_not_validation=True),ensure_ascii=False,indent=2));print(m)
for r in held:print(r['id'],r['target'],r['prediction'].get('ai_points'),r['prediction'].get('standing_confirmed'),r['prediction']['status'])
