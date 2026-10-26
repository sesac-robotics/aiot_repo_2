"""Separate causal stable-Y boundary evidence from ordered-attempt support.

A short/poorly shaped Y attempt is not removed just because stability is late.
If a stable entry is never observed, its boundary is withheld, not filled in.
"""
import numpy as np
from ytw_ready_fix_v2 import YTWTracker as ReadyFixed
from live_observation import mask_frame_border
VERSION='ytw-causal-Y-boundary-refinement-v4-candidate'
class YTWTracker(ReadyFixed):
 def __init__(self,phase_source,view='front',protocol='upright_Y_T_W_v1',dwell=5,max_step=.10,stable_steps=2):
  if not np.isfinite(max_step) or max_step<=0 or stable_steps<1:raise ValueError('Positive motion/steps required')
  self.raw_source=phase_source;self.max_step=float(max_step);self.stable_steps=stable_steps
  self.previous_pose=None;self.previous_pose_tick=None;self.stable_count=0;self.onset_tick=None;self.step=None;self.boundary_trace=[]
  super().__init__(self._source,view,protocol,dwell)
 def _clear_motion(self,clear_onset=False):
  self.previous_pose=None;self.previous_pose_tick=None;self.stable_count=0;self.step=None
  if clear_onset:self.onset_tick=None
 def _reset(self,reason):
  super()._reset(reason);self._clear_motion(clear_onset=True)
 def _source(self,points):
  label,ready=self.raw_source(points)
  eligible=label=='Y' and ready
  if not eligible:
   self._clear_motion(clear_onset=not self.active)
   return label,ready
  shoulder=points[[11,12]].mean(0);hip=points[[23,24]].mean(0);scale=float(np.linalg.norm(shoulder-hip))
  normalized=None if scale<=1 else (points[[13,14,15,16]]-shoulder)/scale
  self.step=None
  if normalized is not None and self.previous_pose is not None and self.previous_pose_tick==self.last_tick-1:
   self.step=float(np.max(np.linalg.norm(normalized-self.previous_pose,axis=1)))
  self.stable_count=self.stable_count+1 if self.step is not None and self.step<=self.max_step else 0
  self.previous_pose=normalized;self.previous_pose_tick=self.last_tick
  if self.onset_tick is None and self.stable_count>=self.stable_steps:
   # Report the actual confirmation tick. Never backdate to the first of the
   # motion samples or use future T/W observations to shift the boundary.
   self.onset_tick=int(self.last_tick)
  return label,ready
 def push(self,tick,xy,width=640,height=480):
  safe=mask_frame_border(xy,width,height)
  if (self.last_tick is not None and tick!=self.last_tick+1) or not np.isfinite(safe[[11,12,13,14,15,16,23,24]]).all():
   self._clear_motion(clear_onset=not self.active)
  return super().push(tick,xy,width,height)
 def _complete(self,width,height):
  onset=self.onset_tick
  event=super()._complete(width,height)
  if event is None:return None
  start=event['start_tick'];win=event['auto_phase_windows'];raw={k:list(v) for k,v in win.items()}
  event['version']=VERSION;event['coarse_phase_windows']=raw
  valid=onset is not None and start<=onset<=start+win['Y'][1]
  event['Y_boundary_evidence']=dict(confirmed=valid,confirmed_tick=onset if valid else None,
        max_step_torso_ratio=self.max_step,stable_steps=self.stable_steps,causal=True,backdated=False,
        quality_grade=None,reason='causal_stable_Y_tick' if valid else 'stable_Y_boundary_unobserved')
  if valid:
   offset=onset-start;event['samples']=event['samples'][offset:];event['start_tick']=onset
   event['auto_phase_windows']={k:[a-offset,b-offset] for k,(a,b) in raw.items()}
   event['auto_phase_windows']['Y'][0]=0
   dense=np.array([p for t,p in event['samples']]);event['observation_coverage']=float(np.isfinite(dense[:,[11,12,13,14,15,16,23,24]]).all((1,2)).mean())
   # Dense missing slots stay NaN. Recompute the temporal observed-slot measure.
   event['temporal_slot_coverage']=float(np.isfinite(dense[:,[11,12,13,14,15,16,23,24]]).all((1,2)).mean())
  else:
   # Preserve counted ordered-attempt evidence separately from an uncertain Y
   # boundary. No valid auto-window contract or score is authorized for this.
   event['auto_phase_windows']=None
  self.events[-1]={k:v for k,v in event.items() if k!='samples'}
  return event
 def status(self):
  d=super().status();d.update(version=VERSION,Y_stable_tick=self.onset_tick,Y_last_step=self.step,Y_boundary_not_quality=True);return d
