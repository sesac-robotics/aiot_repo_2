"""팀 V2의 전처리/시퀀스/다중 label/표시 정책을 그대로 연결합니다."""
from pathlib import Path
import logging
import math
import os
import json
from collections import Counter

import numpy as np

from app.analysis.posture_state import posture_state as empty_result
from app.analysis.registry import get_exercise_definition
from .vendor.classifier import PostureClassifier, SequenceSession
from .vendor.feedback_policy import FeedbackPolicy

# AIFit 등록 이름과 실제 인계 폴더 이름 사이의 명시적 매핑입니다.
MODEL_EXERCISES = {
    'squat': 'squats', 'push_up': 'pushups', 'ytw_raise': 'ytw',
    'plank': 'elbow_plank', 'dumbbell_shoulder_press': 'shoulder_press',
}


def landmark_array(keypoints):
    """기존 이미지 좌표를 ID 순서로 배치합니다. 없는 관절을 채우지 않습니다."""
    if len(keypoints) != 33:
        return None
    try:
        indexed = {point['id']: point for point in keypoints}
        if set(indexed) != set(range(33)):
            return None
        array = np.asarray([[indexed[i][key] for key in ('x', 'y', 'z', 'visibility')]
                            for i in range(33)], dtype=np.float32)
        # Presence는 학습 특징에 없으므로 추가 필터링/보간에 사용하지 않습니다.
        return array if np.isfinite(array).all() else None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def json_safe(value):
    """비유한 값과 NumPy scalar를 JSON에 그대로 내보내지 않습니다."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError('자세 결과는 JSON 값만 포함해야 합니다.')




class PosturePredictor:
    """하나의 자세 워커 전용. 가중치는 재사용하고 문맥별 시퀀스/피드백은 폐기합니다."""
    def __init__(self, root, architecture='cnn', device='cpu', model_factory=PostureClassifier):
        if architecture not in ('cnn', 'gru'):
            raise ValueError('POSTURE_MODEL_ARCHITECTURE는 cnn 또는 gru여야 합니다.')
        if device not in ('cpu', 'cuda'):
            raise ValueError('POSTURE_MODEL_DEVICE는 cpu 또는 cuda여야 합니다.')
        self.root = Path(root).expanduser()
        self.architecture, self.device = architecture, device
        self._factory = model_factory
        self._models = {}
        self.context = None
        self.sequence = self.policy = None

    def reset(self):
        if self.sequence is not None:
            self.sequence.reset()
        if self.policy is not None:
            self.policy.reset()
        self.context = self.sequence = self.policy = None

    def prepare(self, context):
        if self.context == context:
            return
        self.reset()
        if context is None:
            return
        get_exercise_definition(context.exercise_name)
        name = MODEL_EXERCISES[context.exercise_name]
        if name not in self._models:
            model = self._factory(name, self.architecture, root=self.root, device=self.device)
            self._models[name] = model
            if not model.loaded:
                logging.getLogger('uvicorn.error').warning(
                    '[Posture] model=%s_%s unavailable reason=%s', name, self.architecture, model.load_error)
        model = self._models[name]
        self.sequence = SequenceSession(model)
        # 디스크 오류도 호출자에서 unavailable로 격리하며 Rule-based 경로와 공유하지 않습니다.
        self.policy = FeedbackPolicy(root=self.root)
        self.context = context

    def push(self, context, keypoints, timestamp, frame_size):
        self.prepare(context)
        if self.sequence is None:
            return empty_result('inactive', 'no_active_exercise')
        model = self.sequence.classifier
        if not model.loaded:
            return empty_result('unavailable', 'model_unavailable:' + str(model.load_error))
        width, height = frame_size
        array = landmark_array(keypoints)
        before = dict(buffer_frames=len(self.sequence.frames),
                      last_seen=self.sequence.last_seen, last_sample=self.sequence.last_sample,
                      aspect=self.sequence.aspect, subject_token=self.sequence.subject_token)
        # SequenceSession의 원래 missing_or_invalid_pose 정책으로 버퍼를 초기화합니다.
        raw = self.sequence.push(array, timestamp, width / height,
                                 subject_token=(context.session_id, context.exercise_id, context.revision))
        if not raw.get('updated', True):
            return None  # 생략한 샘플은 신규 예측/피드백 투표가 아닙니다.
        result = self.policy.apply(raw, timestamp)
        self._debug(context, keypoints, array, timestamp, frame_size, before, raw, result)
        reason = raw.get('reason')
        if raw.get('model_outputs') is not None:
            status = 'ready'
        elif reason in ('warming_up', 'stream_gap', 'aspect_or_orientation_changed', 'subject_changed'):
            status = 'warming_up'
        elif reason in ('missing_or_invalid_pose', 'insufficient_pose_or_invalid_shape',
                        'sampling_interval_mismatch', 'nonmonotonic_timestamp'):
            status = 'insufficient_pose'
        else:
            status = 'unavailable'
        output = empty_result(status, reason)
        output.update(classification=raw, display_feedback=result['display_feedback'],
                      is_correct=True if raw['status'] == 'NORMAL' else False if raw['status'] == 'ERROR' else None,
                      predicted_at=timestamp if status == 'ready' else None,
                      buffer_frames=raw.get('buffer_frames', 0))
        scores = (raw.get('model_outputs') or {}).get('error_types', {})
        feedbacks = dict(zip(raw.get('error_types', []), raw.get('feedbacks', [])))
        # sigmoid는 다중 label 점수입니다. 보정된 정확도나 Form Score로 변환하지 않습니다.
        output['posture_errors'] = [dict(code=name, label=name, confidence=scores.get(name),
                                         confidence_kind='uncalibrated_sigmoid_not_accuracy',
                                         feedback=feedbacks.get(name)) for name in raw.get('error_types', [])]
        return json_safe(output)

    def _debug(self, context, keypoints, array, timestamp, frame_size, before, raw, result):
        """진단 실패는 추론/피드백에 영향을 주지 않습니다. 원본 영상은 저장하지 않습니다."""
        if os.getenv('MODEL_DEBUG', '1').strip().lower() not in ('1', 'true'):
            return
        try:
            model = self.sequence.classifier
            counts = Counter(p.get('id') for p in keypoints if isinstance(p, dict))
            joints = []
            for ident in model.required:
                point = next((p for p in keypoints if isinstance(p, dict) and p.get('id') == ident), {})
                visibility = point.get('visibility')
                finite = all(isinstance(point.get(k), (int, float)) and math.isfinite(point[k])
                             for k in ('x', 'y', 'z', 'visibility'))
                reason = ('missing_or_duplicate' if counts[ident] != 1 else
                          'nonfinite_or_missing_value' if not finite else
                          'low_visibility' if visibility < .3 else None)
                joints.append(dict(id=ident, visibility=visibility if isinstance(visibility, (int, float))
                                   and math.isfinite(visibility) else None, valid=reason is None, reason=reason))
            reset_reason = raw.get('reason') if len(self.sequence.frames) == 0 else None
            if before['last_seen'] is not None and timestamp - before['last_seen'] > 1:
                reset_reason = reset_reason or 'stream_gap'
            if before['aspect'] is not None and abs(frame_size[0] / frame_size[1] - before['aspect']) > .01:
                reset_reason = reset_reason or 'aspect_or_orientation_changed'
            intervals = np.diff(self.sequence.times)
            row = dict(session_id=context.session_id, exercise_id=context.exercise_id,
                       revision=context.revision, exercise_name=context.exercise_name,
                       timestamp=timestamp, model_name=raw.get('model_name'), device=self.device,
                       frame_size=list(frame_size), aspect=frame_size[0] / frame_size[1],
                       input_shape=list(array.shape) if array is not None else None,
                       keypoint_count=len(keypoints), required_joints=joints,
                       missing_joint_ids=[i for i in range(33) if counts[i] == 0],
                       duplicate_joint_ids=[i for i, count in counts.items() if count > 1],
                       buffer_before=before['buffer_frames'], buffer_after=len(self.sequence.frames),
                       required_frames=model.length, reset_reason=reset_reason,
                       sample_interval_max=float(intervals.max()) if len(intervals) else None,
                       sample_interval_median=float(np.median(intervals)) if len(intervals) else None,
                       classification=raw, display_feedback=result['display_feedback'])
            logging.getLogger('uvicorn.error').info('[ModelDebug] %s', json.dumps(
                json_safe(row), ensure_ascii=False, allow_nan=False))
        except Exception:
            logging.getLogger('uvicorn.error').exception('[Posture] model diagnostic failed')
