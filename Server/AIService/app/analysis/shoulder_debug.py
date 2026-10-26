"""SHOULDER_PRESS_DEBUG=1에서만 동작하는 읽기 전용 진단 출력."""

import json
import logging
import math
import os
import time


def context_fields(context):
    return {
        name: getattr(context, name, None)
        for name in ("session_id", "exercise_id", "exercise_name", "revision")
    }


def session_fields(manager):
    if manager is None:
        return {
            "session_manager_configured": False,
            "gate_reason": "session_manager_missing",
        }
    snapshot = manager.get_snapshot()
    session, exercise = snapshot["session"], snapshot["exercise"]
    reason = (
        "session_not_running"
        if session["session_status"] != "running"
        else (
            "exercise_missing"
            if exercise is None
            else (
                "exercise_not_running"
                if exercise["exercise_status"] != "running"
                else (
                    "other_exercise_selected"
                    if exercise["exercise_name"] != "dumbbell_shoulder_press"
                    else "shoulder_press_ready_for_dispatch"
                )
            )
        )
    )
    return {
        "session_manager_configured": True,
        "session_id": session["session_id"],
        "session_status": session["session_status"],
        "exercise": exercise,
        "gate_reason": reason,
    }


def _json_safe(value):
    # 무효 표본의 NaN/Infinity도 유효 JSON으로 기록하되 측정값을 만들어내지 않습니다.
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class ShoulderPressDebug:
    def __init__(self, *, clock=time.monotonic):
        # 서버 시작/객체 생성 시 읽습니다. 환경변수 변경 후 서버를 재시작하세요.
        self.enabled = os.getenv("SHOULDER_PRESS_DEBUG") == "1"
        self._clock = clock
        self._last = {}
        self._changes = {}

    def emit(self, kind, payload, *, force=False, change=None):
        if not self.enabled:
            return False
        try:
            now = self._clock()
            changed = change is not None and self._changes.get(kind) != change
            if (
                not force
                and not changed
                and now - self._last.get(kind, -math.inf) < 1.0
            ):
                return False
            data = payload() if callable(payload) else payload
            logging.getLogger("uvicorn.error").info(
                "[ShoulderPressDebug] %s",
                json.dumps(
                    _json_safe({"kind": kind, **data}),
                    ensure_ascii=False,
                    allow_nan=False,
                    default=str,
                ),
            )
            self._last[kind] = now
            if change is not None:
                self._changes[kind] = change
            return True
        except Exception:
            # 진단 출력 실패로 기존 판정/영상 게시 흐름을 바꾸지 않습니다.
            return False
