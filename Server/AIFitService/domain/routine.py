"""운동 루틴의 순수 업무 규칙입니다. HTTP 상태 결정은 라우터가 담당합니다."""

import json
import math

from Server.AIFitService.domain.exercise_names import canonical_exercise_name


class RoutineError(ValueError):
    """루틴 부재(missing)와 잘못된 저장 값(invalid)을 구분합니다."""

    def __init__(self, *, kind: str, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


def day_exercises(plan_json: str, day_no: int) -> list[dict]:
    """저장된 여러 버전의 주간 루틴을 정규화하고 선택한 날의 목표를 검증합니다.

    Args:
        plan_json: 저장소에서 읽은 JSON 문자열입니다. 원본은 변경하지 않습니다.
        day_no: 월요일 1부터 일요일 7까지의 번호입니다.

    Returns:
        운동명, 세트, 횟수, 유지 시간(초), 순서가 포함된 새 목록입니다.
        시간 운동의 reps=1은 기존 DB의 NOT NULL 계약을 위한 저장 값입니다.

    Raises:
        RoutineError: 선택한 날이 없거나 목표 값이 잘못되었습니다.
    """
    if type(day_no) is not int or not 1 <= day_no <= 7:
        raise RoutineError(
            kind="invalid", detail="day_no는 1~7 정수여야 합니다."
        )
    try:
        plan = json.loads(plan_json)
    except (TypeError, json.JSONDecodeError):
        raise RoutineError(
            kind="invalid", detail="저장된 루틴 JSON 형식이 잘못되었습니다."
        )

    day_plan = None
    if isinstance(plan, dict):
        days = plan.get("days")
        if isinstance(days, list):
            day_plan = next(
                (
                    item
                    for item in days
                    if isinstance(item, dict) and item.get("day_no") == day_no
                ),
                None,
            )
        elif isinstance(days, dict):
            day_plan = days.get(str(day_no)) or days.get(f"day_{day_no}")
        if day_plan is None and plan.get("day_no") == day_no:
            day_plan = plan
        if day_plan is None:
            day_plan = plan.get(f"day_{day_no}") or plan.get(f"day{day_no}")
    elif isinstance(plan, list):
        day_plan = next(
            (
                item
                for item in plan
                if isinstance(item, dict) and item.get("day_no") == day_no
            ),
            None,
        )

    exercises = (
        day_plan.get("exercises") if isinstance(day_plan, dict) else None
    )
    if not isinstance(exercises, list) or not exercises:
        raise RoutineError(
            kind="missing",
            detail=f"선택한 7일 루틴에 Day {day_no} 운동이 없습니다.",
        )

    normalized = []
    for index, item in enumerate(exercises, start=1):
        if not isinstance(item, dict):
            raise RoutineError(
                kind="invalid", detail="루틴 운동 항목 형식이 잘못되었습니다."
            )
        raw_name = item.get("exercise_name")
        name = (
            canonical_exercise_name(raw_name)
            if isinstance(raw_name, str)
            else raw_name
        )
        sets = item.get("sets")
        reps = item.get("reps")
        duration = item.get("target_duration_sec", item.get("duration_sec"))
        if not isinstance(name, str) or not name.strip():
            raise RoutineError(
                kind="invalid", detail="루틴에 운동 이름이 없습니다."
            )
        if isinstance(sets, bool) or not isinstance(sets, int) or sets < 1:
            raise RoutineError(
                kind="invalid",
                detail=f"{name}: 목표 세트 수가 올바르지 않습니다.",
            )
        if duration is not None:
            try:
                duration = float(duration)
            except (TypeError, ValueError):
                raise RoutineError(
                    kind="invalid",
                    detail=f"{name}: 목표 유지 시간이 올바르지 않습니다.",
                )
            if (
                isinstance(
                    item.get("target_duration_sec", item.get("duration_sec")),
                    bool,
                )
                or not math.isfinite(duration)
                or duration <= 0
            ):
                raise RoutineError(
                    kind="invalid",
                    detail=f"{name}: 목표 유지 시간은 0보다 커야 합니다.",
                )
            reps = (
                1  # DB schema requires reps; duration is the AI Server target.
            )
        elif isinstance(reps, bool) or not isinstance(reps, int) or reps < 1:
            raise RoutineError(
                kind="invalid",
                detail=f"{name}: 목표 반복 횟수가 올바르지 않습니다.",
            )
        order_no = item.get("order_no", index)
        if (
            isinstance(order_no, bool)
            or not isinstance(order_no, int)
            or order_no < 1
        ):
            raise RoutineError(
                kind="invalid", detail=f"{name}: 운동 순서가 올바르지 않습니다."
            )
        normalized.append(
            {
                "exercise_name": name.strip(),
                "sets": sets,
                "reps": reps,
                "target_duration_sec": duration,
                "order_no": order_no,
            }
        )
    return normalized
