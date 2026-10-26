"""카메라를 열지 않고 팀 실행에 필요한 파일과 두 Python 환경을 점검합니다."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    """현재 작업 위치와 무관하게 저장소 상대 경로 및 모델 무결성을 검사합니다."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score-python", type=Path, default=REPO / ".venv-score/bin/python"
    )
    parser.add_argument(
        "--correction-python",
        type=Path,
        default=REPO / ".venv-correction/bin/python",
    )
    args = parser.parse_args()
    failures = []
    checks = {}
    frozen = REPO / "checkpoints/squat-20261025/payload/runtime"
    required = [
        REPO / "GUI/FitHelper/workout_window.ui",
        REPO / "Server/AIFitDB/AIFit.sql",
        frozen / "mediapipe-pilot/models/heavy.task",
        REPO / "Server/PostureAnalysis/posture-error-training/outputs/best.pt",
    ]
    required += list(
        (REPO / "Server/PostureAnalysis/posture-error-training/multi").glob(
            "*/outputs/best.pt"
        )
    )
    for path in required:
        if not path.is_file():
            failures.append("필수 파일 없음: " + str(path.relative_to(REPO)))
    full = REPO / "Server/ExerciseEvaluation/models/pose_landmarker_full.task"
    receipt = json.loads(full.with_suffix(".source.json").read_text())
    if (
        not full.is_file()
        or hashlib.sha256(full.read_bytes()).hexdigest() != receipt["sha256"]
    ):
        failures.append("Full 관절 모델 파일 또는 SHA-256 불일치")
    code = "import importlib,importlib.metadata,json; names=__import__('sys').argv[1:]; [importlib.import_module(n) for n in names]; print(json.dumps({'imports':names,'python':__import__('sys').version}))"
    for kind, python, imports in [
        (
            "score",
            args.score_python,
            [
                "numpy",
                "cv2",
                "mediapipe",
                "sklearn",
                "PyQt6.QtWidgets",
                "fastapi",
                "uvicorn",
                "jwt",
                "openai",
                "multipart",
            ],
        ),
        (
            "correction",
            args.correction_python,
            [
                "numpy",
                "cv2",
                "mediapipe.python.solutions.pose",
                "torch",
                "sklearn",
            ],
        ),
    ]:
        try:
            result = subprocess.run(
                [str(python.absolute()), "-c", code, *imports],
                capture_output=True,
                text=True,
                timeout=45,
            )
            if result.returncode:
                failures.append(
                    kind + " 환경 import 실패: " + result.stderr[-1500:]
                )
            else:
                checks[kind] = json.loads(result.stdout.splitlines()[-1])
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append(kind + " 환경 실행 실패: " + str(exc))
    print(
        json.dumps(
            {
                "repository": str(REPO),
                "files_checked": len(required) + 1,
                "environments": checks,
                "failures": failures,
                "camera_opened": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
