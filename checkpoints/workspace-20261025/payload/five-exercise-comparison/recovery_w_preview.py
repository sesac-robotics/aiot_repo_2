"""동일 W 개발 사례의 기존·보완 단계와 항목 점수를 관절 영상으로 비교한다."""

import json
import pathlib

import cv2
import joblib
import numpy as np

import audit
import render
import recovery_w_phase
import three_items
import ytw_partial_trial
from recovery_prepare import ROOT


def run() -> dict:
    """같은 구간의 W 항목 0~2등급 전후 영상을 만들고 전체 디코딩한다.

    Returns:
        dict: 기존·실험 AI/규칙 등급과 영상. YTW 총점이 아닌 W 항목만
        표시하며 개발에 참고한 사례이므로 독립 시험이 아니다.

    Raises:
        OSError: 모델·캐시·원본·출력 접근 실패.
        ValueError: 실험 지문·영상 디코딩·캐시 연결 불일치.
        RuntimeError: 영상 표시 작업의 온도·메모리 보호 위반.
    """
    cv2.setNumThreads(1)
    phase = audit.read_json(ROOT / "W_phase_trial.json")
    shape = audit.read_json(ROOT / "W_shape_trial.json")
    comparison = audit.read_json(ROOT / "comparison.json")
    sha = audit.digest(ROOT / "comparison.json")
    if (
        phase["comparison_sha256"] != sha
        or shape["comparison_sha256"] != sha
        or shape["phase_trial_sha256"]
        != audit.digest(ROOT / "W_phase_trial.json")
    ):
        raise ValueError("W preview experiments are stale")
    identifier = "qevd_00105254"
    row = next(
        r
        for r in audit.read_json(ROOT / "input_plan.json")["records"]
        if r["id"] == identifier
    )
    cache = next(
        r["extraction"]["cache"]
        for r in audit.read_json(ROOT / "extraction.json")["records"]
        if r["id"] == identifier
    )
    detail = next(r for r in phase["records"] if r["id"] == identifier)
    before = next(
        r
        for e in comparison["exercises"]
        if e["exercise"] == "ytw"
        for r in e["fixed_predictions"]
        if r["id"] == identifier
    )
    calibrated = next(
        r["after"] for r in shape["fixed_predictions"] if r["id"] == identifier
    )
    ai = float(
        joblib.load(ROOT / "W_phase_trial_model.joblib").predict(
            [detail["feature"]]
        )[0]
    )
    with np.load(cache) as data:
        seq = data["image_xy"].copy()
    cap = cv2.VideoCapture(
        row["video_path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
    )
    w, h = cap.get(3), cap.get(4)
    cap.release()
    seq = three_items.mask_image_border(seq, w, h)
    records = []
    for stage, a, r, detected in [
        (
            "before",
            before["ai_levels"][2],
            before["rule_levels"][2],
            ytw_partial_trial.detect(seq),
        ),
        ("candidate", ai, calibrated, recovery_w_phase.detect(seq)),
    ]:
        folder = ROOT / "W_previews" / stage
        folder.mkdir(parents=True, exist_ok=True)
        link = folder / "primary.npz"
        if link.exists() and link.resolve() != pathlib.Path(cache).resolve():
            raise ValueError("W preview cache changed")
        if not link.exists():
            link.symlink_to(cache)
        video = render.overlay(
            row,
            dict(
                cache_folder=str(folder),
                ai_score=a,
                rule_score=r,
                display_unit="/2 W",
                rule_details=dict(
                    evidence=dict(
                        phase_detection=detected,
                        w_shape=dict(morphology=dict(grade=r)),
                    )
                ),
            ),
        )
        cap = cv2.VideoCapture(str(video))
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if stage == "candidate" and count == 22:
                cv2.imwrite(str(folder / "preview_frame.jpg"), frame)
            count += 1
        expected = int(cap.get(7))
        cap.release()
        if count != expected or count == 0:
            raise ValueError("W preview decode failed")
        records.append(
            dict(
                stage=stage,
                path=str(video),
                sha256=audit.digest(video),
                frames=count,
                ai_grade_0_2=float(a),
                rule_grade_0_2=int(r),
                W_frames=detail[
                    "before_frames" if stage == "before" else "after_frames"
                ],
            )
        )
    report = dict(
        comparison_sha256=sha,
        phase_trial_sha256=audit.digest(ROOT / "W_phase_trial.json"),
        shape_trial_sha256=audit.digest(ROOT / "W_shape_trial.json"),
        candidate_ai_model_sha256=audit.digest(
            ROOT / "W_phase_trial_model.joblib"
        ),
        records=records,
        unit="W_item_grade_0_2",
        whole_ytw_total_score=None,
        development_case_informed_design=True,
        independent_test=False,
        all_frames_decoded=True,
        browser_playback_verified=False,
    )
    audit.save_json(ROOT / "W_preview_manifest.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
