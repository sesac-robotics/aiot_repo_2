from pathlib import Path
import sys,json,hashlib,collections
import numpy as np,joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
O=Path(__file__).resolve().parent;P=O.parent/'squat-priority-20261023';S=O.parent/'squat-relabel-expanded-20261023';r=json.loads((P/'comparison.json').read_text());h=next(h for h in r['heads'] if h['item']=='knee_inward');key='knee_inward';cases=[row for row in r['feature_rows'] if row['levels'].get(key) is not None and key in row['features'] and key in row['new_features']]
def q(v):return [float(a) for a in np.percentile(v,[10,50,90])]
def phase(x):
 valid=np.isfinite(x[:,[23,24,25,26,27,28]]).all((1,2));a=x[valid];hip=a[:,[23,24],1].mean(1);bottom=hip>=np.percentile(hip,75);ratio=abs(a[:,25,0]-a[:,26,0])/np.maximum(1,abs(a[:,27,0]-a[:,28,0]));hw=abs(a[:,24,0]-a[:,23,0]);kr=abs(a[:,25,0]-a[:,26,0])/np.maximum(1,hw)
 
 feet=np.isfinite(a[:,[31,32]]).all((1,2));take=bottom&feet
 if take.sum()>=2:
  toe=abs(a[take,31,0]-a[take,32,0]);kv=abs(a[take,25,0]-a[take,26,0])/np.maximum(1,toe);av=abs(a[take,27,0]-a[take,28,0])/np.maximum(1,toe);foot=q(kv)+q(av)+[1.]
 else:foot=[float('nan')]*6+[0.]
 return q(ratio[bottom])+q(kr[bottom])+foot
for row in cases:
 path=(P if row['origin']=='priority_normal_controls' else S)/'observations'/row['id']/'front.npz';z=np.load(path);x=z['image_xy'].copy();bad=(x[:,:,0]<0)|(x[:,:,0]>=z['width'])|(x[:,:,1]<0)|(x[:,:,1]>=z['height']);x[bad]=np.nan;row['phase_features']=row['new_features'][key]['x']+phase(x)
def fit(rows,which):
 y=[x['levels'][key] for x in rows];c=collections.Counter(y);m=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),ExtraTreesRegressor(n_estimators=120,min_samples_leaf=2,max_features=1.,random_state=23,n_jobs=2));return m.fit([x[which] if which=='phase_features' else x['new_features'][key]['x'] for x in rows],y,extratreesregressor__sample_weight=[1/c[v] for v in y])
def met(ps,k):
 y=np.array([p['target'] for p in ps]);v=np.array([p[k] for p in ps]);e=y<2;n=y==2
 return dict(n=len(y),error_n=int(sum(e)),normal_n=int(sum(n)),missed_errors=int(sum((v>=1.5)&e)),normal_flagged=int(sum((v<1.5)&n)),MAE=float(np.mean(abs(v-y))),error_MAE=float(np.mean(abs(v[e]-y[e]))),balanced_MAE=float(np.mean([np.mean(abs(v[y==g]-g)) for g in sorted(set(y))])))
ps=[]
for f in h['folds']:
 train=[x for x in cases if x['person'] in f['train_people']];held=[x for x in cases if x['id'] in f['held_ids']];assert not {x['person'] for x in train}&{x['person'] for x in held};bm=fit(train,'new');am=fit(train,'phase_features')
 for x in held:ps.append(dict(id=x['id'],person=x['person'],fold=f['fold'],target=x['levels'][key],before=float(bm.predict([x['new_features'][key]['x']])[0]),after=float(am.predict([x['phase_features']])[0])))
b=met(ps,'before');a=met(ps,'after');print(b,a,flush=True);print([x for x in ps if x['target']<2],flush=True)
train=[x for x in cases if x['split']=='train'];val=[x for x in cases if x['split']=='validation'];m=fit(train,'phase_features');prior=joblib.load(P/'candidate.joblib')['heads'][key]['model'];vp=[dict(id=x['id'],person=x['person'],target=x['levels'][key],before=float(prior.predict([x['new_features'][key]['x']])[0]),after=float(m.predict([x['phase_features']])[0])) for x in val];vb=met(vp,'before');va=met(vp,'after');improved=a['missed_errors']<b['missed_errors'] and a['normal_flagged']<=b['normal_flagged'] and a['error_MAE']<=b['error_MAE'] and va['missed_errors']<=vb['missed_errors'] and va['normal_flagged']<=vb['normal_flagged']
joblib.dump(dict(model=m,feature_version='bottom-toe-context-v1',train_ids=[x['id'] for x in train],validation_people=sorted({x['person'] for x in val}),runtime_approved=False,development_improved=improved),O/'toe-candidate.joblib')
(O/'toe-comparison.json').write_text(json.dumps(dict(before=b,after=a,improved=improved,predictions=ps,folds=h['folds'],fixed_before=vb,fixed_after=va,fixed_predictions=vp,cases=cases,source_digest=r['source_digest'],criteria_unchanged=True,new_training_labels=0,independent_final_test=False,interpretation='기존 개발 검증의 누락 사례를 보고 특징을 설계한 추가 실험. 새 사람에 대한 독립 성능은 아님'),ensure_ascii=False,indent=2))
