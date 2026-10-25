"""모델·웹 프레임워크에 의존하지 않는 세션 자세 투영의 기본값입니다."""


def posture_state(status: str, reason: str) -> dict[str, object]:
    """미평가 사유를 포함한 화면 초기 상태를 반환합니다. 확률과 정오 판정은 None으로 구분합니다."""
    return dict(
        status=status,
        reason=reason,
        is_correct=None,
        form_score=None,
        posture_errors=[],
        predicted_at=None,
        classification=None,
        display_feedback=None,
    )
