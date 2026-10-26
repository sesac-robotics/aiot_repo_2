"""Pixels only. No scores, labels, source IDs or manual phase windows as inputs."""
import sys
import numpy as np
from common import B
sys.path.insert(0,str(B/'five-exercise-comparison'));sys.path.insert(0,str(B/'team-posture-scoring/Server/AIService'))
import geometry,three_items,plank_item_observation
import side_height
from app.scoring import held_geometry
import importlib.util
spec=importlib.util.spec_from_file_location('standing_height_features',B/'output/squat-return-20261023/features.py');height_features=importlib.util.module_from_spec(spec);spec.loader.exec_module(height_features)

def quantile(x):
 x=np.asarray(x);x=x[np.isfinite(x)];return np.percentile(x,[5,10,25,50,75,90,95]).tolist() if len(x) else None

def temporal(seq,joints):
 if len(seq)<10 or np.isfinite(seq[:,joints]).all((1,2)).mean()<.8:return None
 hip=seq[:,[23,24]].mean(1);scale=np.linalg.norm(seq[:,[11,12]].mean(1)-hip,axis=1)
 a=(seq[:,joints]-hip[:,None,:])/np.maximum(scale[:,None,None],1)
 a=a[np.linspace(0,len(a)-1,16).astype(int)].flatten()
 # No imputation of missing pose frames for this project candidate.
 return a.tolist() if np.isfinite(a).all() else None

def describe(ex,seq,view,duration,baseline=None,width=None,height=None,*,baseline_verified=False,baseline_end=None,segment_start=None):
 if ex in ['squats','pushups','shoulder']:
  notes=three_items.describe(ex,seq,seq if ex=='squats' and view=='front' else None)
  for n in notes:n['bank']=view
  if ex=='squats':
   side=view in ['side','side_oblique']
   # Side clips may start mid-squat. Never calibrate them from their first
   # flexed frames or from a later return in the attempt being evaluated.
   side_reference=side and baseline_verified and baseline is not None and baseline_end is not None and segment_start is not None and np.isfinite([baseline_end,segment_start]).all() and baseline_end<=segment_start
   allowed=view in ['front','front_oblique'] or side_reference
   f=side_height.extract(seq,baseline,width,height,height_features) if side_reference else height_features.extract(seq,baseline,width,height) if allowed else None
   hold='side_requires_verified_past_standing_reference' if side and not side_reference else 'standing_reference_or_return_unobservable'
   notes[1]=dict(item='standing_height_return',bank='height_v4',feature=f['x'] if f else None,eligible=f is not None,holds=[] if f else [hold])
   if side:
    notes[1].update(feature_version='standing-return-height-v4-side-v2',reference_verified=bool(side_reference),reference_end_seconds=baseline_end,segment_start_seconds=segment_start,view=view,phase_source=f.get('phase_source','observed_hip_excursion') if f else None)
   if view=='front':
    valid=np.isfinite(seq[:,[23,24,25,26,27,28]]).all((1,2));ratio=abs(seq[:,25,0]-seq[:,26,0])/np.maximum(abs(seq[:,27,0]-seq[:,28,0]),1);q=quantile(ratio[valid]);notes[2]=dict(item='knee_gap',bank='front_gap_v1',feature=q,eligible=valid.mean()>=.8 and q is not None,holds=[])
   else:
    for i in (2,3):notes[i]['eligible']=False
  if ex=='shoulder':notes[3]['bank']='front_seated'
  return notes
 if ex=='plank':
  a=plank_item_observation.assess(seq,duration,[True]*4);channels=geometry.channels(seq);notes=[]
  for i,r in enumerate(a['items']):
   f=quantile(channels[r['side']+'_'+plank_item_observation.CHANNELS[i]]);notes.append(dict(item=r['criterion_id'],bank='side_video',eligible=r['eligible'] and f is not None,feature=f,holds=r['reasons']))
  c=held_geometry.plank_clearance(seq);notes[2].update(bank='side_support_clearance_v1',feature=c['feature'],eligible=c['eligible'],holds=c['holds'])
  return notes
 if ex=='ytw':
  arm=temporal(seq,[11,12,13,14,15,16,23,24]);trunk=temporal(seq,[11,12,23,24]);return [dict(item=item,bank='upright_temporal_v1',feature=trunk if i==3 else arm,eligible=(trunk if i==3 else arm) is not None,holds=[]) for i,item in enumerate(['y_reach','t_reach','w_shape','upright_trunk_hold'])]
 raise ValueError(ex)
