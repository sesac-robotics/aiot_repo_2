"""Item-local unscored geometry with one coherent, confidence-selected side.
No person identifiers, grades, provider strings or test statistics are inputs.
"""
import sys
from pathlib import Path
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');sys.path.insert(0,str(B/'five-exercise-comparison'));import geometry
SCHEMA='pushup_coherent_item_geometry_v1'
def select_side(seq,confidence,joints):
 choices=[]
 for offset in [0,1]:
  js=[j+offset for j in joints];valid=np.isfinite(seq[:,js]).all((1,2))
  if len(seq)<10 or valid.mean()<.8:continue
  reliability=np.minimum(confidence[:,js,0],confidence[:,js,1]).mean(1)
  choices.append((float(np.mean(np.where(valid,reliability,0))),offset,valid))
 return max(choices,key=lambda r:(r[0],-r[1]))[1:] if choices else None

def describe(seq,confidence):
 output=[]
 def add(item,joints,make):
  result=select_side(seq,confidence,joints);f=None;side=None
  if result is not None:
   side,valid=result;f=make(side,valid)
  if f is not None and not np.isfinite(f).all():f=None
  output.append(dict(item=item,bank='side_semantic_roi_v1',feature=f,eligible=f is not None,holds=[] if f is not None else ['required_item_geometry_not_observed'],side=None if side is None else ['left','right'][side],side_selection='mean_visibility_presence_on_required_coherent_joints',features_contain_no_rule_grades=True))
 def trunk(o,v):
  s,h,a=[seq[:,j+o] for j in [11,23,27]];axis=a-s;length=np.linalg.norm(axis,axis=1);v=v&(length>1)
  if v.mean()<.8:return None
  d=abs(axis[:,0]*(h-s)[:,1]-axis[:,1]*(h-s)[:,0])/np.maximum(length**2,1);deficit=(180-geometry.angle_deg(s,h,a))/180
  return [float(np.percentile(d[v],50)),float(np.percentile(d[v],90)),float(np.percentile(deficit[v],90))]
 def descent(o,v):
  angle=geometry.angle_deg(*[seq[:,j+o] for j in [11,13,15]])/180;return np.percentile(angle[v],[5,10,25,75]).tolist()
 def extension(o,v):
  angle=geometry.angle_deg(*[seq[:,j+o] for j in [11,13,15]])/180;n=max(3,round(len(seq)*.15));tail=v[-n:]
  return np.percentile(angle[-n:][tail],[10,50,90]).tolist() if tail.mean()>=.8 else None
 def support(o,v):
  sh,el,w,hip,knee,ank,toe=[seq[:,j+o] for j in [11,13,15,23,25,27,31]];floor=toe-w;flen=np.linalg.norm(floor,axis=1);leg=np.linalg.norm(hip-knee,axis=1)+np.linalg.norm(knee-ank,axis=1);v=v&(flen>1)&(leg>1)
  if v.mean()<.8:return None
  normal=np.stack([floor[:,1],-floor[:,0]],axis=1);normal*=np.where(normal[:,1]<0,1,-1)[:,None];clearance=((knee-w)*normal).sum(1)/np.maximum(flen*leg,1);angle=geometry.angle_deg(sh,el,w);top=v&(angle>=np.percentile(angle[v],70));ka=geometry.angle_deg(hip,knee,ank)/180
  if top.sum()<3:return None
  return [float(np.median(clearance[top])),float(np.percentile(clearance[top],10)),float(np.percentile(clearance[v],10)),float(np.percentile(ka[top],10))]
 add('trunk_line',[11,23,27],trunk);add('descent_range',[11,13,15],descent);add('return_extension',[11,13,15],extension);add('support',[11,13,15,23,25,27,31],support)
 # Shoulder lowering relative to the hand/foot support line provides an
 # independently measured depth cue when elbow localization is inaccurate.
 # These are geometric inputs, never a threshold-generated project grade.
 result=select_side(seq,confidence,[11,15,23,31]);f=None;side=None
 if result is not None:
  side,v=result;sh,w,hip,toe=[seq[:,j+side] for j in [11,15,23,31]];axis=toe-w;length=np.linalg.norm(axis,axis=1);torso=np.linalg.norm(sh-hip,axis=1);v=v&(length>1)&(torso>1)
  if v.mean()>=.8:
   normal=np.stack([axis[:,1],-axis[:,0]],axis=1);normal*=np.where(normal[:,1]<0,1,-1)[:,None];gap=((sh-w)*normal).sum(1)/np.maximum(length*torso,1);q=np.percentile(gap[v],[10,50,90]);f=[float(q[0]),float(q[1]),float(q[2]),float(q[2]-q[0])]
 output.append(dict(item='descent_range',bank='side_chest_displacement_roi_v1',feature=f,eligible=f is not None,holds=[] if f is not None else ['required_shoulder_support_geometry_unobserved'],side=None if side is None else ['left','right'][side],features_contain_no_rule_grades=True))
 return output
