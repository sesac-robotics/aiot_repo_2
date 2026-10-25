"""Causal Y entry hold candidate; quality morphology is not a readiness rule."""
import numpy as np
from ytw_ready_fix_v2 import YTWTracker as ReadyFixedTracker
VERSION='ytw-causal-stable-Y-entry-v3-candidate'
class YTWTracker(ReadyFixedTracker):
 def __init__(self,phase_source,view='front',protocol='upright_Y_T_W_v1',dwell=5,max_step=.08):
  if not np.isfinite(max_step) or max_step<=0:raise ValueError('Positive finite step limit required')
  self.raw_source=phase_source;self.max_step=float(max_step);self.previous_pose=None;self.step=None;self.previous_pose_tick=None
  super().__init__(self._source,view,protocol,dwell)
 def _reset(self,reason):
  super()._reset(reason);self.previous_pose=None;self.previous_pose_tick=None;self.step=None
 def _source(self,points):
  label,ready=self.raw_source(points)
  sh=points[[11,12]].mean(0);hip=points[[23,24]].mean(0);scale=float(np.linalg.norm(sh-hip))
  normalized=None if scale<=1 else (points[[13,14,15,16]]-sh)/scale
  self.step=None
  if normalized is not None and self.previous_pose is not None and self.previous_pose_tick==self.last_tick-1:
   self.step=float(np.max(np.linalg.norm(normalized-self.previous_pose,axis=1)))
  self.previous_pose=normalized;self.previous_pose_tick=self.last_tick
  # A movement hold gate only. No elbow angle, Y quality/severity, or arm-down
  # return is required. Missing/nonconsecutive samples cannot provide stability.
  if not self.active:
   ready=bool(ready and self.step is not None and self.step<=self.max_step)
  return label,ready
 def _complete(self,width,height):
  event=super()._complete(width,height)
  if event:
   event['version']=VERSION;event['Y_entry_gate']=dict(max_step_torso_ratio=self.max_step,coordinate='arm_joint_relative_to_shoulder_normalized_by_torso',causal=True,quality_grade=None)
   self.events[-1].update(version=VERSION,Y_entry_gate=event['Y_entry_gate'])
  return event
 def status(self):
  d=super().status();d.update(version=VERSION,Y_entry_max_step=self.max_step,Y_entry_last_step=self.step);return d
