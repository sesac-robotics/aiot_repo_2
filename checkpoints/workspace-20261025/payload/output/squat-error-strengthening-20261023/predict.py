"""Offline diagnostic only: preserve deployed heads and do not emit a 100-point total."""
from pathlib import Path
from functools import lru_cache
import sys,joblib,numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O.parent/'live-webcam-20261023'))
from item_observation import observed_features
from features import stance_context
@lru_cache(maxsize=1)
def bundle():return joblib.load(O/'candidate.joblib')
def predict(xy,width,height):
 x=np.asarray(xy,float).copy();bad=(x[:,:,0]<0)|(x[:,:,0]>=width)|(x[:,:,1]<0)|(x[:,:,1]>=height);x[bad]=np.nan
 core=observed_features(x,width,height);stance=stance_context(x);out=[]
 for key in ['knee_inward','trunk_balance']:
  if key not in core:
   out.append(dict(item=key,ai_grade=None,rule_grade=None,status='required_item_observation_missing'));continue
  group=bundle()['heads'][key];context=key=='knee_inward' and stance is not None and group['stance'] is not None;head=group['stance'] if context else group['core'];obs=dict(core[key]);route='optional_ankle_width_context' if context else 'core_without_ankles'
  if context:obs=dict(x=obs['x']+stance['x'],scalar=stance['scalar'])
  ai=float(np.clip(head['model'].predict([obs['x']])[0],0,2));v=obs['scalar'];g,p=head['boundaries'];h=head['higher'];rule=2 if (v>=g if h else v<=g) else 1 if (v>=p if h else v<=p) else 0
  out.append(dict(item=key,ai_grade=ai,rule_grade=rule,route=route,status='offline_candidate_only'))
 return dict(items=out,runtime_approved=False,total=None,gpt_used=False)
