"""Height-return neural comparison with person-separated early stopping."""
import copy,collections,warnings
import numpy as np,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import ConvergenceWarning
from common import B,O,read,save,sha,status
warnings.filterwarnings('ignore',category=ConvergenceWarning)

def fit_epoch(X,y,groups,epochs=None):
 unique=sorted(set(groups));validation_group=unique[-1];tr=groups!=validation_group;va=~tr
 if epochs is not None:tr=np.ones(len(y),bool);va=~tr
 scaler=StandardScaler().fit(X[tr]);m=MLPRegressor(hidden_layer_sizes=(32,16),activation='relu',solver='adam',learning_rate_init=.002,batch_size=min(32,int(tr.sum())),max_iter=1,shuffle=True,random_state=241024);best=None;bestloss=float('inf');patience=0;history=[]
 pairs=collections.Counter(zip(y[tr],groups[tr]));grade_groups=collections.Counter(label for label,group in pairs);sample_weights=np.array([1/(grade_groups[label]*pairs[label,group]) for label,group in zip(y[tr],groups[tr])]);sample_weights*=len(sample_weights)/sample_weights.sum()
 for epoch in range(1,(epochs or 80)+1):
  m.partial_fit(scaler.transform(X[tr]),y[tr]/2,sample_weight=sample_weights)
  if epochs is None:
   pred=np.clip(m.predict(scaler.transform(X[va]))*2,1,2);loss=float(np.mean(abs(pred-y[va])));history.append(dict(epoch=epoch,validation_MAE=loss))
   if epoch>=5 and loss<bestloss-1e-5:bestloss=loss;best=copy.deepcopy(m);selected=epoch;patience=0
   else:patience+=1
   if epoch>=15 and patience>=12:break
  else:selected=epoch;best=m
 return dict(model=best,scaler=scaler,selected_epoch=selected,epochs_run=epoch,history=history)
def predict(b,X):return np.clip(b['model'].predict(b['scaler'].transform(X))*2,1,2)
def main():
 rows=read(O/'score-inputs.frozen.json');rs=[]
 for r in rows:
  for n,g in zip(r['items'],r['levels']):
   if n['item']=='standing_height_return' and n['eligible'] and g is not None:rs.append(dict(id=r['id'],person=r['person'],group=r['group'],split=r['split'],x=n['feature'],target=g))
 tr=[r for r in rs if r['split']=='train'];va=[r for r in rs if r['split']=='validation'];X=np.array([r['x'] for r in tr]);y=np.array([r['target'] for r in tr]);groups=np.array([r['group'] for r in tr]);folds=[];predictions=[]
 for ti,vi in GroupKFold(n_splits=4).split(X,y,groups):
  assert not set(groups[ti])&set(groups[vi]);chosen=fit_epoch(X[ti],y[ti],groups[ti]);b=fit_epoch(X[ti],y[ti],groups[ti],epochs=chosen['selected_epoch']);pred=predict(b,X[vi]);folds.append(dict(selected_epoch=chosen['selected_epoch'],epochs_run=chosen['epochs_run'],history=chosen['history'],outer_validation_groups=sorted(set(groups[vi]))));predictions.extend(dict(id=tr[i]['id'],group=tr[i]['group'],target=float(y[i]),prediction=float(p)) for i,p in zip(vi,pred));status('return_neural_fold',epochs=chosen['epochs_run'])
 selected=round(np.median([f['selected_epoch'] for f in folds]));b=fit_epoch(X,y,groups,epochs=selected);b.update(supported_grades=[1,2],feature_version='standing-return-height-v4',training_ids=[r['id'] for r in tr],production_replacement=False);joblib.dump(b,O/'return-neural-candidate.joblib')
 if va:vp=[dict(id=r['id'],group=r['group'],target=r['target'],prediction=float(p)) for r,p in zip(va,predict(b,np.array([r['x'] for r in va])))]
 else:vp=[]
 from train_scores import metrics
 save('return-neural-comparison.json',dict(group_cv=metrics(predictions),fixed_development_validation=metrics(vp),folds=folds,full_training_epochs=selected,training_n=len(tr),validation_n=len(va),predictions=predictions,validation_predictions=vp,supported_grades=[1,2],severe_grade_zero_supported=False,final_test_used=False,production_replacement=False,model_sha256=sha(O/'return-neural-candidate.joblib'),note='Manual person-separated stopping split inside training; no random-frame early stopping. Temporal height summary inputs, not an end-to-end video network.'))
 status('return_neural_complete',epochs=selected)
if __name__=='__main__':main()
