"""화면 표시 이름을 시스템 운동 식별자로 변환합니다. 외부 I/O는 수행하지 않습니다."""

_EXERCISE_NAMES = {
    "plank": "plank",
    "플랭크": "plank",
    "squat": "squat",
    "스쿼트": "squat",
    "dumbbell_shoulder_press": "dumbbell_shoulder_press",
    "덤벨숄더프레스": "dumbbell_shoulder_press",
    "push_up": "push_up",
    "푸쉬업": "push_up",
    "푸시업": "push_up",
    "ytw_raise": "ytw_raise",
    "ytw레이즈": "ytw_raise",
}


def canonical_exercise_name(name: str) -> str:
    """Map Korean display names and canonical keys to the AI Server key."""
    if not isinstance(name, str):
        return name
    display_name = name.strip()
    lookup_key = "".join(display_name.casefold().split())
    return _EXERCISE_NAMES.get(lookup_key, display_name)


def canonicalize_routine_exercises(routine: dict | None) -> dict | None:
    """Normalize exercise names in a routine response, including legacy plans."""
    if not isinstance(routine, dict):
        return routine
    days = routine.get("days")
    if not isinstance(days, list):
        return routine
    for day in days:
        if not isinstance(day, dict):
            continue
        exercises = day.get("exercises")
        if not isinstance(exercises, list):
            continue
        for exercise in exercises:
            if isinstance(exercise, dict) and isinstance(
                exercise.get("exercise_name"), str
            ):
                exercise["exercise_name"] = canonical_exercise_name(
                    exercise["exercise_name"]
                )
    return routine
