from pathlib import Path
import json,numpy as np,joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
O=Path(__file__).resolve().parent;r=json.loads((O/'toe-comparison.json').read_text());cases=r['cases'];folds=r['folds']
def vector(x):return x['phase_features'][14:]
def fit(rows,C):
 m=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=C,class_weight='balanced',max_iter=2000,random_state=23));return m.fit([vector(x) for x in rows],[int(x['levels']['knee_inward']==2) for x in rows])
def predict(m,x):return float(2*m.predict_proba([vector(x)])[0,1])
def met(ps,key):
 y=np.array([p['target'] for p in ps]);v=np.array([p[key] for p in ps]);e=y<2;n=y==2
 return dict(n=len(y),error_n=int(sum(e)),normal_n=int(sum(n)),missed_errors=int(sum(e&(v>=1.5))),normal_flagged=int(sum(n&(v<1.5))),MAE=float(np.mean(abs(v-y))),error_MAE=float(np.mean(abs(v[e]-y[e]))),balanced_MAE=float(np.mean([np.mean(abs(v[y==g]-g)) for g in sorted(set(y))])))
def choose(train):
 options=[];groups=[x['person'] for x in train]
 for C in [.01,.1,1.]:
  ps=[]
  for tr,te in GroupKFold(min(3,len(set(groups)))).split(train,groups=groups):
   rows=[train[i] for i in tr]
   if len({x['levels']['knee_inward'] for x in rows})<2:continue
   m=fit(rows,C)
   for i in te:ps.append(dict(target=train[i]['levels']['knee_inward'],pred=predict(m,train[i])))
  mt=met(ps,'pred');options.append(dict(C=C,score=mt['balanced_MAE']+.5*mt['missed_errors']/max(1,mt['error_n'])+.25*mt['normal_flagged']/max(1,mt['normal_n']),metrics=mt))
 return min(options,key=lambda x:(x['score'],x['C']))['C'],options
preds=[];audit=[]
for fold in folds:
 train=[x for x in cases if x['person'] in fold['train_people']];held=[x for x in cases if x['id'] in fold['held_ids']];C,options=choose(train);m=fit(train,C)
 for x in held:
  p=next(p for p in r['predictions'] if p['id']==x['id']);preds.append(dict(id=x['id'],person=x['person'],fold=fold['fold'],target=x['levels']['knee_inward'],before=p['before'],after=predict(m,x)))
 audit.append(dict(fold=fold['fold'],train_people=fold['train_people'],held_people=fold['held_people'],C=C,inner_selection=options))
b=met(preds,'before');a=met(preds,'after');train=[x for x in cases if x['split']=='train'];val=[x for x in cases if x['split']=='validation'];C,options=choose(train);m=fit(train,C);prior=joblib.load(O.parent/'squat-priority-20261023/candidate.joblib')['heads']['knee_inward']['model'];vp=[dict(id=x['id'],person=x['person'],target=x['levels']['knee_inward'],before=float(prior.predict([x['new_features']['knee_inward']['x']])[0]),after=predict(m,x)) for x in val];vb=met(vp,'before');va=met(vp,'after');approved=a['missed_errors']<b['missed_errors'] and a['normal_flagged']<=b['normal_flagged'] and a['error_MAE']<=b['error_MAE'] and va['missed_errors']<=vb['missed_errors'] and va['normal_flagged']<=vb['normal_flagged'];joblib.dump(dict(model=m,feature_version='bottom-foot-logistic-v1',train_ids=[x['id'] for x in train],C=C,development_improved=approved,runtime_approved=False),O/'linear-candidate.joblib');(O/'linear-comparison.json').write_text(json.dumps(dict(before=b,after=a,improved=approved,predictions=preds,folds=audit,fixed_before=vb,fixed_after=va,fixed_predictions=vp,independent_final_test=False,criteria_unchanged=True,new_training_labels=0),ensure_ascii=False,indent=2));print(b,a,approved);print([p for p in preds if p['target']<2])
