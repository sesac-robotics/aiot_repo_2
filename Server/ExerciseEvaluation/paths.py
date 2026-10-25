"""저장소 기준 보존 소스 경로와 평가별 격리 실행 환경을 구성합니다."""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FROZEN = REPO / "checkpoints/squat-20261025/payload/runtime"
SCORE_LIVE = FROZEN / "output/live-webcam-20261023"
SCORE_APP = FROZEN / "team-posture-scoring/Server/AIService"
CORRECTION = REPO / "Server/PostureAnalysis/posture-error-training"


def worker_environment(kind):
    """모델별 원래 모듈을 별도 프로세스에서 찾도록 경로를 명시합니다."""
    env = os.environ.copy()
    locations = [str(REPO)]
    locations += (
        [str(CORRECTION)]
        if kind == "correction"
        else [str(SCORE_LIVE), str(SCORE_APP)]
    )
    env["PYTHONPATH"] = os.pathsep.join(locations)
    env["OMP_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    return env
