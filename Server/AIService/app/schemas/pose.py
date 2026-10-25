"""API 요청·응답의 전송 자료형을 정의합니다. 유효한 JSON 형식과 실제 자세 판정의 정확성은 별개입니다."""

from datetime import datetime

from pydantic import BaseModel, Field


class PoseKeypoint(BaseModel):
    id: int
    name: str

    x: float
    y: float
    z: float

    visibility: float | None = None
    presence: float | None = None


class PoseResponse(BaseModel):
    frame_id: int
    processed_at: datetime

    model: str
    person_detected: bool

    keypoints: list[PoseKeypoint] = Field(default_factory=list)
