"""SQUAT_DEBUG=1 전용 읽기 진단. Reps 분석기/임계값/상태를 변경하지 않습니다."""
import json
import logging
import math
import os
import time


class SquatDebug:
    def __init__(self):
        self.enabled = os.getenv('SQUAT_DEBUG','0') == '1'
        self._last_log = self._pipeline_last = -math.inf
        self._change = None

    @staticmethod
    def _finite(value):
        return type(value) in (int,float) and math.isfinite(value)

    def pipeline(self, manager, stage):
        if not self.enabled:
            return
        try:
            snapshot = manager.get_snapshot()
            exercise = snapshot['exercise']
            now = time.monotonic()
            if exercise is None or exercise['exercise_name'] != 'squat' or now-self._pipeline_last < 1:
                return
            self._pipeline_last = now
            logging.getLogger('uvicorn.error').info('[SquatDebug] %s',json.dumps(dict(
                kind='pipeline',stage=stage,analyzer_called=False,
                session_id=snapshot['session']['session_id'],session_status=snapshot['session']['session_status'],
                exercise_id=exercise['exercise_id'],rep_count=exercise['rep_count']),ensure_ascii=False))
        except Exception:
            pass  # 진단 실패가 영상/이벤트 처리로 전파되지 않게 합니다.

    def emit(self,manager,context,analyzer,points,*,frame_id,timestamp,frame_size,previous_phase,
             previous_timestamp,previous_cycle_started,reason,analyzer_called,event,manager_called,accepted):
        if not self.enabled:
            return
        try:
            snapshot = manager.get_snapshot()
            exercise = snapshot['exercise']
            if (getattr(context,'exercise_name',None) != 'squat'
                    and (exercise is None or exercise['exercise_name'] != 'squat')):
                return
            active = getattr(analyzer,'_analyzer',analyzer)
            cfg = getattr(active,'config',None)
            if cfg is None:
                from app.analysis.exercises.squat import SquatConfig
                cfg = SquatConfig()
            phase = getattr(active,'phase',None)
            required = (23,24,25,26,27,28)
            joints = {}
            indexed = {}
            for ident in required:
                found = [p for p in points if isinstance(p,dict) and p.get('id') == ident]
                p = found[0] if len(found) == 1 else {}
                error = 'missing_or_duplicate' if len(found) != 1 else None
                if error is None and any(not self._finite(p.get(k)) or not 0 <= p[k] <= 1 for k in ('x','y')):
                    error = 'coordinates_invalid'
                if error is None and (not self._finite(p.get('visibility')) or not cfg.min_visibility <= p['visibility'] <= 1):
                    error = 'visibility_invalid_or_low'
                if error is None and p.get('presence') is not None and (not self._finite(p['presence']) or not cfg.min_presence <= p['presence'] <= 1):
                    error = 'presence_invalid_or_low'
                joints[str(ident)] = {k:p.get(k) if self._finite(p.get(k)) else None for k in ('x','y','visibility','presence')}
                joints[str(ident)].update(valid=error is None,invalid_reason=error)
                if error is None:
                    indexed[ident] = p
            angles = {}
            from app.analysis.exercises.squat import knee_angle
            for side,ids in (('left',(23,25,27)),('right',(24,26,28))):
                if frame_size is not None and all(i in indexed for i in ids):
                    try:
                        angles[side] = knee_angle(*(indexed[i] for i in ids),frame_size)
                    except (ValueError,TypeError,OverflowError):
                        angles[side] = None
                else:
                    angles[side] = None
            valid = all(v['valid'] for v in joints.values()) and all(v is not None for v in angles.values())
            invalid = [f'joint_{i}_{v["invalid_reason"]}' for i,v in joints.items() if not v['valid']]
            difference = abs(angles['left']-angles['right']) if all(v is not None for v in angles.values()) else None
            if difference is not None and difference > cfg.max_leg_difference_deg:
                invalid.append('left_right_angle_difference_exceeded');valid=False
            if not invalid and not valid:
                invalid.append('angle_or_frame_size_invalid')
            mean = sum(angles.values())/2 if valid else None
            now = time.monotonic()
            change = (getattr(context,'revision',None),phase,tuple(invalid),analyzer_called)
            if event is None and change == self._change and now-self._last_log < 1:
                return
            self._last_log,self._change = now,change
            candidate = getattr(active,'_candidate',None)
            since = getattr(active,'_candidate_since',None)
            condition = dict(standing=mean >= cfg.standing_enter_deg,descending=mean <= cfg.standing_exit_deg,
                             bottom=mean <= cfg.bottom_enter_deg,ascending=mean >= cfg.bottom_exit_deg) if mean is not None else {}
            cycle_duration = (timestamp-previous_cycle_started
                              if self._finite(timestamp) and self._finite(previous_cycle_started) else None)
            row = dict(kind='reps',frame_id=frame_id,frame_timestamp=timestamp,
                session_id=getattr(context,'session_id',snapshot['session']['session_id']),
                exercise_id=getattr(context,'exercise_id',exercise['exercise_id'] if exercise else None),
                revision=getattr(context,'revision',None),session_status=snapshot['session']['session_status'],
                analyzer_called=analyzer_called,analyzer_name=type(active).__name__,
                previous_phase=previous_phase,current_phase=phase,candidate_phase=candidate,
                candidate_samples=getattr(active,'_candidate_samples',0),
                candidate_duration_sec=timestamp-since if self._finite(timestamp) and self._finite(since) else None,
                frame_gap_sec=timestamp-previous_timestamp if self._finite(timestamp) and self._finite(previous_timestamp) else None,
                cycle_duration_sec=cycle_duration,
                cycle_duration_valid=(cfg.min_rep_duration_sec <= cycle_duration <= cfg.max_rep_duration_sec)
                    if cycle_duration is not None else None,
                joints=joints,left_knee_deg=angles['left'],right_knee_deg=angles['right'],
                mean_knee_deg=mean,left_right_difference_deg=difference,input_valid=valid,invalid_reasons=invalid,
                conditions=condition,thresholds=vars(cfg),reason=reason,
                rep_completed=event is not None,event_id=event.event_id if event else None,
                manager_called=manager_called,event_accepted=accepted,
                event_rejected_reason=reason if event is not None and accepted is not True else None,
                rep_count=exercise['rep_count'] if exercise else None)
            logging.getLogger('uvicorn.error').info('[SquatDebug] %s',json.dumps(row,ensure_ascii=False,allow_nan=False))
        except Exception:
            pass
