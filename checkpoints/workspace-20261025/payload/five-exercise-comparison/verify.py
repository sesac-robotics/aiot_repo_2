"""원본 보존·최신 관절 캐시·표시 영상·개발 비교의 연결을 검사한다.

이 검사는 파일·계산 무결성 검사이며 라벨 정확도나 모델 성능 인증이 아니다.
"""

import collections
import pathlib
import re
import subprocess
import sys
import urllib.request

import cv2
import numpy as np

import audit
import extract
from resources import check_resources


def run() -> dict:
    """현재 결과의 계산·연결 검사를 수행하고 JSON 보고서를 저장한다.

    Returns:
        dict: 검사별 통과 여부, 캐시·영상 수, 최종 시험 미완료 상태.

    Raises:
        OSError: 필수 파일 또는 로컬 HTTP 서버에 접근할 수 없는 경우.
        ValueError: 캐시 누락·버전 불일치 또는 원본 파일이 변경된 경우.
        RuntimeError: 온도·메모리·전원 보호 조건을 위반한 경우.
        subprocess.SubprocessError: 테스트 실패 또는 제한 시간 초과.
    """
    checked = audit.read_json(audit.OUT / "audit.json")
    original_unchanged = all(
        audit.digest(audit.SOURCE / (name + ".json")) == fingerprint
        for name, fingerprint in checked["source_json_sha256"].items()
    )
    if not original_unchanged or not checked["passed"]:
        raise ValueError("Original source or structural audit failed")
    manifest = audit.read_json(audit.OUT / "development_manifest.json")
    selected = [
        row
        for row in manifest
        if row["extraction_ready"]
        and row["original_split"] in ("train", "validation")
    ]
    caches = []
    for row in selected:
        views = (
            ["side", "front"]
            if row["exercise"] == "squats"
            else (
                ["side"]
                if row["exercise"] in ("pushups", "plank")
                else ["primary"]
            )
        )
        for view in views:
            path = audit.OUT / "heavy" / row["id"] / (view + ".json")
            metadata = audit.read_json(path)
            if metadata["config"]["schema"] != extract.SCHEMA:
                raise ValueError("Incomplete cache migration: " + row["id"])
            with np.load(path.with_suffix(".npz")) as data:
                if str(data["fingerprint"]) != metadata["fingerprint"]:
                    raise ValueError("Cache fingerprint mismatch")
                frames = data["frame_numbers"]
                if not (np.diff(frames) > 0).all():
                    raise ValueError("Non-monotonic video samples")
            caches.append(metadata)
    overlays = []
    for path in sorted((audit.OUT / "heavy").glob("*/overlay.mp4")):
        check_resources()
        capture = cv2.VideoCapture(
            str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded = 0
        while capture.read()[0]:
            decoded += 1
        capture.release()
        if decoded == 0 or decoded != expected:
            raise ValueError("Overlay decode failed: " + str(path))
        overlays.append(
            dict(
                path=str(path), decoded_frames=decoded, all_frames_decoded=True
            )
        )
    test_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(pathlib.Path(__file__).parent),
            "-p",
            "test_*.py",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    with urllib.request.urlopen(
        "http://127.0.0.1:8767/", timeout=5
    ) as response:
        http_status = response.status
    comparison = audit.read_json(audit.OUT / "comparison.json")
    report = dict(
        structural_audit_passed=True,
        original_six_json_unchanged=True,
        records=checked["records"],
        source_video_paths_checked=checked["source_video_paths_checked"],
        view_caches=len(caches),
        cache_schema=extract.SCHEMA,
        cache_fingerprints_checked=True,
        overlay_checks=overlays,
        http_status=http_status,
        unit_test_suite_passed=True,
        unit_test_count=int(
            re.search(r"Ran (\d+) tests", test_result.stderr).group(1)
        ),
        models_by_status=dict(
            collections.Counter(
                result["model_status"]
                for result in comparison["results"].values()
            )
        ),
        comparison_sha256=audit.digest(audit.OUT / "comparison.json"),
        extraction_seconds_sum=float(sum(r["extract_seconds"] for r in caches)),
        extraction_timing_note_ko="원본 읽기·각 시점 Heavy 초기화·추출·저장을 포함한 기록값 합. CPU 제한 변경과 중단·재개가 있어 통제된 속도 벤치마크가 아님.",
        expert_validated=False,
        independent_final_test_count=0,
        all_five_complete=False,
    )
    audit.save_json(audit.OUT / "verification.json", report)
    return report


if __name__ == "__main__":
    print(run())
