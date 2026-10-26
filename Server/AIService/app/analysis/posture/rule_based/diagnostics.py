"""SQUAT_DEBUG 전용 자세 평가 진단. 기존 판정 결과를 읽기만 합니다."""
import json
import logging
import math
import os
import time


class SquatPostureDebug:
    def __init__(self):
        self.enabled = os.getenv('SQUAT_DEBUG', '0') == '1'
        self._last = -math.inf
        self._change = None

    def emit(self, predictor, context, pose, checks, result, timestamp, analysis_result, keypoints, frame_gap_sec):
        if not self.enabled:
            return
        try:
            code = 'SQUAT_KNEE_VALGUS'
            check = next(c for c in checks if c.code == code)
            gate = predictor.gates.get(code)
            gap_reset = frame_gap_sec is not None and frame_gap_sec > predictor.config.max_frame_gap_sec
            change = (context, predictor.view, result['classification']['phase'],
                      check.state, check.reason, result['reason'], result['is_correct'], gap_reset)
            now = time.monotonic()
            if change == self._change and now-self._last < 1:
                return
            self._last, self._change = now, change
            # 유효성은 실제 PoseGeometry 결과를 읽고 무효 관절도 원본 신뢰도를 기록합니다.
            joints = {}
            raw = {p['id']:p for p in keypoints if isinstance(p,dict) and type(p.get('id')) is int}
            if pose is not None:
                for ident in (23,24,25,26,27,28):
                    point = raw.get(ident, {})
                    joints[str(ident)] = dict(
                        valid=ident in pose.points,
                        invalid_reason=pose.invalid.get(ident, 'missing') if ident not in pose.points else None,
                        **{key: point.get(key) if type(point.get(key)) in (int,float)
                           and math.isfinite(point[key]) else None for key in ('x','y','visibility','presence')})
            cfg = predictor.config
            measured = getattr(predictor.evaluator, 'diagnostics', {})
            row = dict(
                kind='posture', session_id=context.session_id, exercise_id=context.exercise_id,
                timestamp=timestamp, effective_camera_view=predictor.view,
                frame_gap_sec=frame_gap_sec,
                temporal_reset_by_frame_gap=gap_reset,
                camera_view_source='POSTURE_CAMERA_VIEWS.squat' if 'squat' in predictor.views else 'POSTURE_CAMERA_VIEW',
                default_camera_view=predictor.default_view, check_code=code,
                check_enabled=check.state != 'not_applicable',
                check_evaluated=code in result['evaluated_checks'],
                check_pending_reason=check.reason if check.bad is None else None,
                knee_ankle_width_ratio=measured.get('knee_ankle_width_ratio'),
                projection=measured, joints=joints,
                phase=result['classification']['phase'],
                phase_source=result['classification'].get('phase_source'),
                front_motion=result.get('front_motion'),
                analyzer_phase=getattr(analysis_result, 'phase', None),
                analyzer_timestamp=getattr(analysis_result, 'timestamp', None),
                analyzer_exercise_name=getattr(analysis_result, 'exercise_name', None),
                thresholds=dict(min_visibility=cfg.min_visibility, min_presence=cfg.min_presence,
                    valgus_enter_ratio=cfg.valgus_enter_ratio, valgus_exit_ratio=cfg.valgus_exit_ratio,
                    confirm_sec=cfg.confirm_sec, clear_sec=cfg.clear_sec,
                    normal_confirm_sec=cfg.normal_confirm_sec, max_frame_gap_sec=cfg.max_frame_gap_sec),
                temporal_pending_reason=result['reason'],
                error_candidate_samples=gate.samples if gate else 0,
                error_candidate_duration_sec=timestamp-gate.since if gate and gate.since is not None else None,
                normal_candidate_samples=predictor.normal_samples,
                normal_candidate_duration_sec=timestamp-predictor.normal_since if predictor.normal_since is not None else None,
                evaluated_checks=result['evaluated_checks'], unevaluated_checks=result['unevaluated_checks'],
                not_applicable_checks=result['not_applicable_checks'],
                status=result['status'], reason=result['reason'], is_correct=result['is_correct'],
            )
            logging.getLogger('uvicorn.error').info('[SquatDebug] %s', json.dumps(row, ensure_ascii=False, allow_nan=False))
        except Exception:
            pass  # 진단 기록 실패가 평가/반복 처리에 영향을 주지 않게 합니다.
