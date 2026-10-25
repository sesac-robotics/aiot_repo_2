"""AIFit이 발급한 정수 ID를 사용하는 내부 세션 API 계약."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class ExerciseStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exercise_name: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ]
    order_no: Annotated[int, Field(strict=True, gt=0)]
    target_reps: Annotated[int, Field(strict=True, gt=0)] | None = None
    target_duration_sec: (
        Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)] | None
    ) = None


class SessionSnapshot(BaseModel):
    session_id: int | None
    session_status: Literal["idle", "running", "paused", "finished"]
    started_at: str | None
    ended_at: str | None
    paused_at: str | None
    paused_duration_sec: float
    active_duration_sec: float


class ExerciseSnapshot(BaseModel):
    exercise_id: int
    exercise_name: str
    order_no: int
    target_reps: int | None
    rep_count: int
    exercise_status: Literal["running", "finished"]


class HoldExerciseSnapshot(ExerciseSnapshot):
    current_hold_sec: float
    target_duration_sec: float | None
    hold_status: Literal[
        "waiting", "stabilizing", "holding", "interrupted", "completed"
    ]
    hold_completed: bool


class PostureAnalysisSnapshot(BaseModel):
    # 패키지 원시 분류/피드백의 실제 class mapping을 보존합니다.
    model_config = ConfigDict(extra="allow")
    status: Literal[
        "warming_up",
        "ready",
        "insufficient_pose",
        "unavailable",
        "paused",
        "inactive",
    ]
    reason: str | None = None
    is_correct: bool | None = None
    form_score: float | None = None
    posture_errors: list[dict] = Field(default_factory=list)
    predicted_at: float | None = None
    classification: dict | None = None
    display_feedback: dict | None = None


class SessionLiveResponse(BaseModel):
    session: SessionSnapshot
    exercise: HoldExerciseSnapshot | ExerciseSnapshot | None
    posture_analysis: PostureAnalysisSnapshot | None = None
