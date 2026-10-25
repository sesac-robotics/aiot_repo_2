"""모델을 import하지 않는 Live 자세 상태 기본값."""


def posture_state(status, reason):
    return dict(status=status, reason=reason, is_correct=None, form_score=None,
                posture_errors=[], predicted_at=None, classification=None, display_feedback=None)
