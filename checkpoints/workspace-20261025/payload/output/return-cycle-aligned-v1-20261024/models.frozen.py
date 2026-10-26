"""Small regularized learned models for scarce person-separated criterion labels."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
class ScaledRegressor:
 def __init__(self,name):self.name=name
 def fit(self,X,y,sample_weight=None):
  X=np.asarray(X);self.n_features_in_=X.shape[1];self.scaler=StandardScaler().fit(X,sample_weight=sample_weight);y=np.asarray(y);self.classes_=np.unique(y)
  if len(self.classes_)==1:self.model=None;self.constant=float(y[0]);return self
  if self.name.startswith('ridge'):self.model=Ridge(alpha=float(self.name[5:]));self.model.fit(self.scaler.transform(X),y,sample_weight=sample_weight)
  else:self.model=LogisticRegression(C=float(self.name[5:] if self.name.startswith('grade') else self.name[7:]),max_iter=2000,random_state=241024);self.model.fit(self.scaler.transform(X),y,sample_weight=sample_weight)
  return self
 def predict(self,X):
  if self.model is None:return np.full(len(X),self.constant)
  X=self.scaler.transform(X)
  if self.name.startswith('ridge'):return self.model.predict(X)
  p=self.model.predict_proba(X)
  if self.name.startswith('grade'):
   # First decide normal versus any error, then the observed error severity.
   normal=p[:,list(self.model.classes_).index(2)] if 2 in self.model.classes_ else np.zeros(len(X))
   errors=self.model.classes_<2
   grades=self.model.classes_[errors][np.argmax(p[:,errors],axis=1)] if errors.any() else np.full(len(X),2)
   return np.where(normal>=.5,2,grades)
  return p@self.model.classes_

class NormalThresholdRegressor:
 """A severity classifier with a cutoff learned only from Train group OOF."""
 def __init__(self,base,normal_threshold):
  self.base=base;self.normal_threshold=float(normal_threshold);self.n_features_in_=base.n_features_in_
  self.project_grade_min=base.project_grade_min;self.project_grade_max=base.project_grade_max
 def predict(self,X):
  p=self.base.model.predict_proba(self.base.scaler.transform(np.asarray(X,float)))
  classes=self.base.model.classes_;normal=p[:,list(classes).index(2)]
  errors=classes<2;grades=classes[errors][np.argmax(p[:,errors],axis=1)]
  return np.where(normal>=self.normal_threshold,2,grades)

class PositiveDeficitClassifier:
 """Train a binary severity classifier whose error risk grows with deficits.
 No zero-grade support is created when Train contains only grades 1 and 2.
 """
 def __init__(self,C=1.):self.C=float(C)
 def fit(self,X,y,sample_weight=None):
  from scipy.optimize import minimize
  from scipy.special import expit
  X=np.asarray(X,float);y=np.asarray(y);self.n_features_in_=X.shape[1];self.classes_=np.unique(y)
  if set(self.classes_)!={1,2}:raise ValueError('Positive deficit model requires observed grades1 and2')
  self.project_grade_min=1;self.project_grade_max=2;w=np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight,float);w=w/w.sum();z=(y<2).astype(float)
  self.scaler=StandardScaler().fit(X,sample_weight=w);a=self.scaler.transform(X)
  def fun(v):
   logit=a@v[:-1]+v[-1];err=expit(logit)-z;loss=np.sum(w*(np.logaddexp(0,logit)-z*logit))+.5*np.sum(v[:-1]**2)/(self.C*len(y));grad=np.r_[a.T@(w*err)+v[:-1]/(self.C*len(y)),np.sum(w*err)];return loss,grad
  fit=minimize(fun,np.zeros(a.shape[1]+1),jac=True,bounds=[(0,None)]*a.shape[1]+[(None,None)],method='L-BFGS-B',options={'maxiter':2000,'ftol':1e-12});
  if not fit.success:raise ValueError(fit.message)
  self.coef_=fit.x[:-1];self.intercept_=float(fit.x[-1]);self.normal_threshold=.5;return self
 def normal_probability(self,X):
  from scipy.special import expit
  return expit(-(self.scaler.transform(np.asarray(X,float))@self.coef_+self.intercept_))
 def predict(self,X):return np.where(self.normal_probability(X)>=self.normal_threshold,2.,1.)

class GroupCalibratedDeficitEnsemble:
 """Average person-held component models; calibrate without a full-data refit.
 The cutoff uses predictions from models trained on the same number of groups
 as the deployed ensemble components, avoiding a full-fit probability shift.
 """
 def __init__(self,members,normal_threshold=.5):
  if not members:raise ValueError('No fitted members')
  self.members=members;self.normal_threshold=float(normal_threshold);self.n_features_in_=members[0].n_features_in_;self.project_grade_min=1;self.project_grade_max=2
 def normal_probability(self,X):return np.mean([m.normal_probability(X) for m in self.members],axis=0)
 def predict(self,X):return np.where(self.normal_probability(X)>=self.normal_threshold,2.,1.)

class MonotoneExtensionEnsemble(GroupCalibratedDeficitEnsemble):
 """Learn extension severity from seven angular deficits, ignoring visibility.
 Visibility gates observation outside the estimator. It is not a severity cue.
 Each component has nonnegative deficit coefficients; more extension cannot
 increase error risk. Only observed grade1/2 support is returned.
 """
 def __init__(self,members,normal_threshold=.5):
  super().__init__(members,normal_threshold)
  if any(m.n_features_in_!=7 for m in members):raise ValueError('Expected seven learned angular deficits')
  self.n_features_in_=8
 @staticmethod
 def deficits(X):
  a=np.asarray(X,float)
  if a.ndim!=2 or a.shape[1]!=8 or not np.isfinite(a).all():raise ValueError('Expected observed degree quantiles and visibility')
  if np.any((a[:,:7]<0)|(a[:,:7]>180)) or np.any((a[:,7]<0)|(a[:,7]>1)):raise ValueError('Invalid angle/visibility units')
  return (180-a[:,:7])/180
 def normal_probability(self,X):
  a=self.deficits(X)
  return np.mean([m.normal_probability(a) for m in self.members],axis=0)
