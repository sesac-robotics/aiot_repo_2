"""직접 확인된 운동 변형의 원본 해시로 후속 반복의 대상 일치를 검사한다."""


def standing_allowed(row: dict, excluded_sources: set[str]) -> bool:
    """서기 숄더 프레스에 앉은 변형 원본이 재진입하는 것을 막는다.

    Args:
        row: 운동 코드·원본 SHA-256·선택적인 관측 변형을 가진 행.
        excluded_sources: 보존된 직접 검토 기록에서 확인된 앉은 원본 해시.

    Returns:
        bool: 현재 대상에 허용되는지 여부. 다른 운동은 그대로 허용한다.
        참여자 이름·제공자 정상 오류 문자열로 변형을 추정하지 않는다.

    Raises:
        KeyError: 운동 코드 또는 원본 해시가 없는 경우.
    """
    return row["exercise"] != "shoulder" or (
        row["video_sha256"] not in excluded_sources
        and row.get("observed_variant")
        not in ("seated_press", "seated_dumbbell_shoulder_press")
    )
