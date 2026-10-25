from pathlib import Path
import sys,json,numpy as np,collections
from sklearn.ensemble import ExtraTreesRegressor
O=Path(__file__).resolve().parent;R=O.parents[1];sys.path.insert(0,str(O.parent/'live-webcam-20261023'))
from runtime_snapshot import prepare_snapshot
runtime,_=prepare_snapshot();sys.path.insert(0,str(runtime));from app.scoring import three_items
from accepted_items import load
sys.path.insert(0,str(O.parent/'squat-priority-20261023'))
from geometry_features import features
_,extract=load();r=json.loads((O/'phase-comparison.json').read_text());rows=r['cases'];f=next(f for f in r['folds'] if 'rehab26_Vasil' in f['held_people']);tr=[a for a in rows if a['person'] in f['train_people']];y=[a['levels']['knee_inward'] for a in tr];c=collections.Counter(y);m=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=2,max_features=1.,random_state=23,n_jobs=2).fit([a['new_features']['knee_inward']['x'] for a in tr],y,sample_weight=[1/c[v] for v in y]);out=[]
for ident in ['Vasil_01','Vasil_02','Vasil_08']:
 old=np.load(O.parent/'squat-relabel-expanded-20261023/observations'/ident/'front.npz');new=np.load(O/(ident+'-image.npz'));a=features(old['image_xy'],float(old['width']),float(old['height']),extract,three_items)[1]['knee_inward'];b=features(new['image_xy'],float(new['width']),float(new['height']),extract,three_items)[1]['knee_inward'];out.append(dict(id=ident,before=float(m.predict([a['x']])[0]),reextracted=float(m.predict([b['x']])[0]),max_feature_delta=float(np.max(abs(np.array(a['x'])-b['x']))),features_before=a,features_reextracted=b))
(O/'extraction-counterfactual.json').write_text(json.dumps(dict(cases=out,train_people=f['train_people'],held_person='rehab26_Vasil',purpose='추출 방식만 바꾼 원인 분리 실험; 정답 관절 오차 확정 아님'),ensure_ascii=False,indent=2));print([(a['id'],a['before'],a['reextracted']) for a in out])
