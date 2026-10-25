from uuid import UUID
from typing import Literal
from datetime import datetime

from pydantic import BaseModel, Field

class SessionStartRequest(BaseModel):
    user_id: UUID
    exercise_type: str
    target_reps = int | None = Field(default=None, ge=1)

class KeypointData(BaseModel):
    id: int
    name: str
    x: float
    y: float
    confidence: float | None = None

class LiveAnalysisResult(BaseModel):
    session_id: UUID
    frame_id: int
    timestamp: datetime

    exercise_type: str
    rep_count: int
    duration_sec: float

    phase: str
    pose_score: float | None = None
    feedback: str = ''
    error_codes: list[str] = Field(default_factory=list)

    person_detected: bool
    keypoints: list[KeypointData] = Field(default_factory=list)

class ExerciseRepRecord(BaseModel):
    rep_no: int
    pose_score: float | None = None
    grade: str | None = None
    error_codes: list[str] = Field(default_factory=list)
    feedback: str = ''

class SessionResult(BaseModel):
    session_id: UUID
    user_id: UUID
    exercise_id: str

    started_at: datetime
    ended_at: datetime

    total_reps: int
    average_score: float | None = None
    grade: str | None = None

    error_counts: dict[str, int] = Field(default_factory=dict)
    feedback: str = ''

    records: list[ExerciseRepRecord] = Field(default_factory=list)