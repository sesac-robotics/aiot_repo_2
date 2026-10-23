"""QEVD 기본 레코드를 보존하며 리뷰 구간의 점수를 독립 실행한다.

AIService 디렉터리에서 python -m app.scoring --help로 사용한다.
서버·카메라·DB·팀 통신 계약을 시작하거나 변경하지 않는다.
"""

import argparse
import hashlib
import json
import logging
import pathlib
import subprocess
import urllib.request
from typing import Any, cast

import numpy as np

from app.scoring import engine, media, report
from app.scoring.sequence import PoseSequence

LOGGER = logging.getLogger(__name__)
HEAVY_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)


def demo_sequence() -> PoseSequence:
    """외부 영상 없이 실행을 확인할 합성 숄더 동작을 생성한다.

    Returns:
        PoseSequence: 640×480 픽셀·30fps 원본의 10Hz 팔 상승·하강 31표본.
        정확도·정상 자세 검증 자료가 아닌 실행용 합성 좌표다.
    """
    seq = np.full((31, 33, 2), np.nan)
    phase = np.sin(np.linspace(0, np.pi, 31))
    for k, p in enumerate(phase):
        seq[k, 11] = [260, 200]
        seq[k, 12] = [380, 200]
        seq[k, 23] = [270, 340]
        seq[k, 24] = [370, 340]
        seq[k, 13] = [220, 250 - 140 * p]
        seq[k, 14] = [420, 250 - 140 * p]
        seq[k, 15] = [200, 290 - 220 * p]
        seq[k, 16] = [440, 290 - 220 * p]
    return PoseSequence(seq, np.arange(31) * 3, 30.0, 640, 480)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> int:
    """요청 검증·추출·점수·HTML을 수행하고 CLI 종료 코드를 반환한다.

    Returns:
        int: 성공 0, 입력·모델·파일·추론 오류 2. 실패 시 오류를 숨기지 않는다.
        --download-heavy는 검증된 공식 모델을 저장하고 즉시 종료한다.

    Raises:
        OSError: 실패 로그조차 쓸 수 없는 환경 등 복구 불가 파일 오류.
    """
    parser = argparse.ArgumentParser(
        description="검토 구간의 5종 AI / Rule 오프라인 점수 비교"
    )
    parser.add_argument(
        "--request",
        type=pathlib.Path,
        help="QEVD 네 필드와 assessment_input 확장 JSON",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="새 결과 폴더; 기존 폴더는 덮어쓰지 않음",
    )
    parser.add_argument(
        "--landmarks", type=pathlib.Path, help="영상 추출 대신 수치 NPZ 입력"
    )
    parser.add_argument("--front-landmarks", type=pathlib.Path)
    parser.add_argument(
        "--heavy-model", type=pathlib.Path, help="MediaPipe Heavy .task 경로"
    )
    parser.add_argument(
        "--delegate",
        choices=["CPU", "GPU"],
        default="CPU",
        help="학습 추출은 GPU; CPU 결과는 별도 검증 필요",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="원본에 관절과 점수를 표시한 MP4 생성; ffmpeg 필요",
    )
    parser.add_argument(
        "--bundle", type=pathlib.Path, default=engine.DEFAULT_BUNDLE
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="합성 입력으로 실행 확인; 정확도 시험 아님",
    )
    parser.add_argument(
        "--download-heavy",
        type=pathlib.Path,
        help="공식 Heavy 모델을 지정한 새 파일에 다운로드",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        if args.download_heavy:
            path = args.download_heavy
            if path.exists():
                raise ValueError("Model download target already exists")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".part")
            try:
                with (
                    urllib.request.urlopen(HEAVY_URL, timeout=60) as response,
                    temporary.open("xb") as output,
                ):
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                if (
                    hashlib.sha256(temporary.read_bytes()).hexdigest()
                    != media.HEAVY_SHA256
                ):
                    raise ValueError(
                        "Official model hash differs from validated Heavy"
                    )
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            return 0
        if args.output is None or (args.request is None and not args.demo):
            parser.error("--output and --request (or --demo) are required")
        seq_front = None
        extraction: dict[str, Any] = {}
        if args.demo:
            if args.request or args.landmarks or args.overlay:
                raise ValueError("Demo cannot mix real inputs or overlay")
            seq = demo_sequence()
            qevd = dict(
                video_path="./synthetic_demo.mp4",
                labels=[],
                labels_descriptive=[],
                split="test",
            )
            options: dict[str, Any] = dict(
                exercise="shoulder", view="front", complete_attempt=True
            )
        else:
            request = json.loads(args.request.read_text(encoding="utf-8"))
            if not isinstance(request, dict) or not all(
                k in request
                for k in (
                    "video_path",
                    "labels",
                    "labels_descriptive",
                    "split",
                    "assessment_input",
                )
            ):
                raise ValueError(
                    "QEVD fields plus assessment_input are required"
                )
            if (
                not isinstance(request["video_path"], str)
                or not isinstance(request["labels"], list)
                or not isinstance(request["labels_descriptive"], list)
                or request["split"] not in ("train", "validation", "test")
            ):
                raise ValueError("Invalid QEVD field types or split")
            qevd = {
                k: request[k]
                for k in ("video_path", "labels", "labels_descriptive", "split")
            }
            options = dict(request["assessment_input"])
            video = (args.request.parent / request["video_path"]).resolve()
            if args.landmarks:
                seq = media.load_landmarks(args.landmarks)
            else:
                if args.heavy_model is None:
                    raise ValueError(
                        "--heavy-model required for video extraction"
                    )
                seq, extraction = media.extract_video(
                    video,
                    args.heavy_model,
                    start_frame=options["start_frame"],
                    end_frame=options["end_frame"],
                    delegate=args.delegate,
                )
            if args.front_landmarks:
                seq_front = media.load_landmarks(args.front_landmarks)
            elif "front_video_path" in options:
                if args.heavy_model is None:
                    raise ValueError("Heavy model required for second video")
                seq_front, extra = media.extract_video(
                    (
                        args.request.parent / options["front_video_path"]
                    ).resolve(),
                    args.heavy_model,
                    start_frame=options["front_start_frame"],
                    end_frame=options["front_end_frame"],
                    delegate=args.delegate,
                )
                extraction["front"] = extra
        allowed = {
            k: options[k]
            for k in (
                "view",
                "paired_view_reviewed",
                "lag_frames",
                "complete_attempt",
                "permitted_items",
                "phase_windows",
                "projection_reviewed",
            )
            if k in options
        }
        scorer = engine.ScoringEngine(args.bundle)
        result = scorer.score(
            options["exercise"], seq, front=seq_front, **allowed
        )
        result.update(
            qevd=qevd,
            extraction=extraction,
            synthetic_input=args.demo,
            output_complete=False,
        )
        args.output.mkdir(parents=True, exist_ok=False)
        result = cast(dict[str, Any], _json_safe(result))
        (args.output / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        media.save_landmarks(args.output / "primary.npz", seq)
        if seq_front is not None:
            media.save_landmarks(args.output / "front.npz", seq_front)
        video_name = None
        if args.overlay:
            video_name = report.overlay(video, seq, result, args.output).name
        report.write_html(result, args.output, video_name)
        result["output_complete"] = True
        temporary = args.output / "result.tmp"
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output / "result.json")
        print(
            json.dumps(
                dict(
                    output=str(args.output.resolve()),
                    ai_score=result["ai_score"],
                    rule_score=result["rule_score"],
                ),
                ensure_ascii=False,
            )
        )
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        LOGGER.error("Posture scoring failed: %s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
