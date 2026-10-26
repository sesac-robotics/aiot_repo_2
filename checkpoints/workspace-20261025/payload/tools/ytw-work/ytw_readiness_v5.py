"""Current observation, pending Y, and completed order are separate evidence.

Adds receipts/status to v4.1 without changing model thresholds or segmentation.
No live app integration, camera, grade model, or automatic approval is performed.
"""
import numpy as np
from ytw_hold_onset_v4 import YTWTracker as BoundaryTracker
from live_observation import mask_frame_border, JOINT_NAMES
VERSION='ytw-observation-readiness-receipts-v5-candidate'
REQUIRED=[11,12,13,14,15,16,23,24]
class YTWTracker(BoundaryTracker):
 def __init__(self,phase_source,view='front',protocol='upright_Y_T_W_v1',dwell=5,max_step=.10,stable_steps=2):
  self._provided_source=phase_source;self._frame=None;self._receipts={};self._ended=False;self._current_Y_streak=0;self._ready_tick=None
  super().__init__(self._capture_source,view,protocol,dwell,max_step,stable_steps)
 def _reset(self,reason):
  super()._reset(reason);self._receipts={}
 def _capture_source(self,points):
  label,ready=self._provided_source(points)
  if self._frame is not None:
   self._frame.update(phase_label=label,ready_class=None if label is None else bool(ready))
   tick=self._frame['tick']
   self._current_Y_streak=(self._current_Y_streak+1 if self._ready_tick==tick-1 else 1) if label=='Y' and ready else 0
   self._ready_tick=tick if label=='Y' and ready else None
  return label,ready
 def push(self,tick,xy,width=640,height=480):
  # Reject malformed timestamps before modifying receipt state.
  if not isinstance(tick,(int,np.integer)) or tick<0 or (self.last_tick is not None and tick<=self.last_tick):
   raise ValueError('Strictly increasing nonnegative integer slots required')
  safe=mask_frame_border(xy,width,height);raw=np.asarray(xy,float)
  missing=[j for j in REQUIRED if not np.isfinite(safe[j]).all()]
  self._frame=dict(tick=int(tick),valid=not missing,missing_joint_ids=missing,
       missing_joints=[JOINT_NAMES[j] for j in missing],
       border_joint_ids=[j for j in missing if np.isfinite(raw[j]).all()],phase_label=None,ready_class=None)
  if missing or (self.last_tick is not None and tick!=self.last_tick+1):
   self._current_Y_streak=0;self._ready_tick=None
  self._ended=False;was_active=self.active;was_expected=self.expected
  event=super().push(tick,xy,width,height)
  if self.active and not was_active:
   self._receipts.update(Y_phase_candidate_at_tick=int(self.samples[0][0]),Y_ready_confirmed_at_tick=int(tick))
  if self.active and was_expected=='T' and self.expected=='W':
   self._receipts.update(T_phase_candidate_at_tick=int(self.windows['T'][0]),T_phase_confirmed_at_tick=int(tick))
  if self.active and self.onset_tick is not None:
   self._receipts['Y_stable_confirmed_at_tick']=int(self.onset_tick)
  return event
 def _complete(self,width,height):
  receipts=dict(self._receipts)
  if self.onset_tick is not None:receipts['Y_stable_confirmed_at_tick']=int(self.onset_tick)
  event=super()._complete(width,height)
  if event is not None:
   receipts.update(W_phase_candidate_at_tick=int(event['coarse_start_tick']+event['coarse_phase_windows']['W'][0]),W_phase_confirmed_at_tick=int(self.last_tick))
   event['candidate_context']=dict(view=self.view,protocol=self.protocol,production_enabled=False)
   event['readiness_receipts']=dict(receipts,ordered_sequence_confirmed_at_tick=int(self.last_tick),
        semantics='candidate and confirmations are observations, not physical onset or severity truth',
        production_enabled=False,grade=None)
   # Keep version of boundary detector intact for timing/geometry comparisons.
   event['readiness_version']=VERSION
   self.events[-1]={k:v for k,v in event.items() if k!='samples'}
  return event
 def finish(self):
  self._ended=True
  return super().finish()
 def status(self):
  result=super().status();f=self._frame or {};observed=bool(f.get('valid'))
  supported=self.view=='front' and self.protocol=='upright_Y_T_W_v1'
  current_Y=observed and f.get('phase_label')=='Y' and f.get('ready_class') is True
  ready=bool(not self._ended and supported and current_Y and self.active and self.expected=='T' and self._current_Y_streak>=self.dwell and 'Y_ready_confirmed_at_tick' in self._receipts)
  stable_ready=ready and self.onset_tick is not None and self.stable_count>=self.stable_steps
  if self._ended:state,message='observation_ended','관측 종료 · 미완료는 실패 등급으로 바꾸지 않습니다.'
  elif not supported:state,message='unsupported_protocol','현재 후보가 지원하지 않는 구도·프로토콜입니다.'
  elif not observed:state,message='observation_wait','관측 대기 · 필요한 팔·몸통 관절이 화면 안쪽에 보여야 합니다.'
  elif f.get('phase_label') not in ('Y','T','W'):state,message='state_model_unavailable','상태 판정 근거가 없어 준비를 보류합니다.'
  elif stable_ready:state,message='Y_ready_and_boundary_confirmed','Y 준비와 안정 시각 확인 · T 전환 대기'
  elif ready:state,message='Y_ready_boundary_pending','Y 순서 준비 확인 · 안정 경계는 확인 중'
  elif current_Y and not self.active:state,message='Y_confirmation_pending','Y 확인 중 · 연속 관측을 기다립니다.'
  elif self.active:state,message='ordered_attempt_in_progress','Y→T→W 순서 관측 중 · 품질 점수와는 별도입니다.'
  else:state,message='waiting_Y','지정된 Y 시작 상태를 기다립니다.'
  result.update(readiness_version=VERSION,current_observation=dict(f),current_state=state,message=message,
       current_Y_ready=ready,current_Y_ready_streak=self._current_Y_streak,current_Y_boundary_confirmed=bool(stable_ready),
       attempt_Y_ready_observed='Y_ready_confirmed_at_tick' in self._receipts,
       attempt_receipts=dict(self._receipts),production_enabled=False,quality_grade=None)
  return result


def scoring_input_hold(event):
 """Admission diagnostics only; never returns true manual/total permissions."""
 holds=[]
 context=event.get('candidate_context',{})
 if context.get('view')!='front' or context.get('protocol')!='upright_Y_T_W_v1':holds.append('unsupported_or_missing_protocol')
 if event.get('complete_attempt_reviewed') is not False:holds.append('automatic_evidence_cannot_grant_manual_complete_approval')
 if event.get('projection_reviewed') is not False:holds.append('projection_approval_not_available')
 if event.get('phase_windows_reviewed') is not None:holds.append('automatic_windows_not_manual_review')
 if not event.get('Y_boundary_evidence',{}).get('confirmed'):holds.append('Y_boundary_unconfirmed')
 windows=event.get('auto_phase_windows')
 samples=event.get('samples',[])
 if not windows or set(windows)!={'Y','T','W'}:holds.append('automatic_phase_windows_unavailable')
 if not samples:holds.append('samples_unavailable')
 else:
  ticks=np.asarray([t for t,p in samples]);xy=np.asarray([p for t,p in samples],float)
  if not np.array_equal(ticks,np.arange(event['start_tick'],event['end_tick']+1)):holds.append('actual_dense_slots_required')
  if xy.shape!=(len(samples),33,2):holds.append('coordinate_shape_mismatch')
  else:
   safe=mask_frame_border(xy,event['width'],event['height']);coverage=float(np.isfinite(safe[:,REQUIRED]).all((1,2)).mean())
   if coverage<.8:holds.append('required_joints_below80percent')
   if not np.allclose(safe,xy,equal_nan=True):holds.append('unmasked_border_coordinates')
   if windows and set(windows)=={'Y','T','W'}:
    malformed=any(not isinstance(w,list) or len(w)!=2 or any(not isinstance(t,int) for t in w) for w in windows.values())
    if malformed:holds.append('phase_window_shape_mismatch')
    elif any(not 0<=a<=b<len(samples) for a,b in windows.values()):holds.append('phase_window_outside_segment')
    elif not windows['Y'][1]<windows['T'][0]<=windows['T'][1]<windows['W'][0]:holds.append('phase_window_order_mismatch')
 return dict(input_contract_eligible=not holds,holds=holds,
       existing_engine_options=dict(view='front',complete_attempt=False,phase_windows=None,projection_reviewed=False),
       full_score_authorized=False,grade=None,scope='input_contract_not_quality_or_model_support')
