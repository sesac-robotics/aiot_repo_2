from pathlib import Path
import json,collections,numpy as np
from sklearn.model_selection import GroupKFold
O=Path(__file__).resolve().parent;prior=json.loads((O.parent/'squat-priority-20261023/comparison.json').read_text());reports=[]
def predict(v,b,h):return 2 if (v>=b[0] if h else v<=b[0]) else 1 if (v>=b[1] if h else v<=b[1]) else 0
def fit(rows,key,mult=1):
 higher=rows[0]['features'][key]['higher'];v=np.array([r['features'][key]['scalar'] for r in rows]);y=np.array([r['levels'][key] for r in rows]);values=np.unique(v);cuts=np.unique(np.r_[values[0]-.01,(values[:-1]+values[1:])/2,values[-1]+.01]);cuts=np.unique(np.quantile(cuts,np.linspace(0,1,min(25,len(cuts)))));c=collections.Counter(y);w=np.array([(mult if a<2 else 1)/c[a] for a in y]);best=None
 for a in cuts:
  for b in cuts:
   if (higher and a<b) or (not higher and a>b):continue
   pred=np.where(v>=a,2,np.where(v>=b,1,0)) if higher else np.where(v<=a,2,np.where(v<=b,1,0));loss=float(np.average(abs(pred-y),weights=w));option=(loss,float(a),float(b))
   if best is None or option<best:best=option
 return list(best[1:]),higher

def metrics(ps):
 y=np.array([p['target'] for p in ps]);v=np.array([p['prediction'] for p in ps]);e=y<2;n=y==2
 return dict(n=len(y),error_n=int(e.sum()),normal_n=int(n.sum()),missed_errors=int(sum((v>=1.5)&e)),normal_flagged=int(sum((v<1.5)&n)),MAE=float(np.mean(abs(v-y))),error_MAE=float(np.mean(abs(v[e]-y[e]))) if any(e) else None,balanced_MAE=float(np.mean([np.mean(abs(v[y==g]-g)) for g in sorted(set(y))])))
for h in prior['heads']:
 key=h['item'];cases=[r for r in prior['feature_rows'] if r['levels'].get(key) is not None and key in r['features'] and key in r['new_features']];before=[];after=[];folds=[]
 for fold in h['folds']:
  train=[r for r in cases if r['person'] in fold['train_people']];held=[r for r in cases if r['id'] in fold['held_ids']];opts=[]
  for mult in [1,2,4]:
   ps=[];groups=[r['person'] for r in train]
   for tr,te in GroupKFold(min(3,len(set(groups)))).split(train,groups=groups):
    b,higher=fit([train[i] for i in tr],key,mult)
    for i in te:ps.append(dict(target=train[i]['levels'][key],prediction=predict(train[i]['features'][key]['scalar'],b,higher)))
   m=metrics(ps);opts.append(dict(weight=mult,score=m['balanced_MAE']+.5*m['missed_errors']/max(1,m['error_n'])+.25*m['normal_flagged']/max(1,m['normal_n']),metrics=m))
  selected=min(opts,key=lambda x:(x['score'],x['weight']))['weight'];bb,higher=fit(train,key);ab,_=fit(train,key,selected)
  for r in held:
   p=dict(id=r['id'],person=r['person'],target=r['levels'][key],fold=fold['fold']);before.append(dict(p,prediction=predict(r['features'][key]['scalar'],bb,higher)));after.append(dict(p,prediction=predict(r['features'][key]['scalar'],ab,higher)))
  folds.append(dict(fold=fold['fold'],train_people=fold['train_people'],held_people=fold['held_people'],weight=selected,before_bounds=bb,after_bounds=ab,inner_results=opts))
 b=metrics(before);a=metrics(after);improved=a['missed_errors']<b['missed_errors'] and a['normal_flagged']<=b['normal_flagged'] and a['error_MAE']<=b['error_MAE'] and a['balanced_MAE']<=b['balanced_MAE'];reports.append(dict(item=key,before=b,after=a,improved=improved,before_predictions=before,after_predictions=after,folds=folds));print(key,b,a,improved,flush=True)
(O/'rule-focus-comparison.json').write_text(json.dumps(dict(reports=reports,criterion_unchanged=True,baseline='Same relative-gap/trunk-change scalar; boundaries fit on each training fold only',runtime_rules_changed=False,independent_final_test=False),ensure_ascii=False,indent=2))
