
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

    keypoints: list[PoseKeypoint] = Field(
        default_factory=list
    )
