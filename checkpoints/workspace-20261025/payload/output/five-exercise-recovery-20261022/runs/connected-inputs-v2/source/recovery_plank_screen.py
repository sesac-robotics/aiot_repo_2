"""보유 QEVD 플랭크 후보를 지지 항목에 필요한 측면 구도 위주로 자동 선별한다."""

import json
import pathlib

import cv2
import numpy as np

import audit
import extract
import external_review
from recovery_prepare import ROOT
from resources import check_resources


def run() -> dict:
    """새 보유 후보 80개의 짧은 관절 표본을 선별하고 상위 8개 검토판을 만든다.

    Returns:
        dict: 관측률·수평 몸통 비율·후보 순위·검토판 경로. 선별 결과는
        직접 검토·자세 등급·실제 접점 확인 또는 학습 가능 판정이 아니다.

    Raises:
        OSError: 보유 원본·대기열·캐시 접근 실패.
        ValueError: 원본 메타데이터·프레임 계약 오류.
        RuntimeError: 자원 보호 조건 위반.
    """
    rows = [
        r
        for r in audit.read_json(ROOT / "seventy_plan.json")["candidates"]
        if r["exercise"] == "plank"
    ]
    (ROOT / "review").mkdir(exist_ok=True)
    results = []
    for row in rows:
        check_resources()
        cap = cv2.VideoCapture(row["path"])
        fps = cap.get(5)
        count = int(cap.get(7))
        cap.release()
        first = max(0, round(count / 2 - fps * 0.5))
        last = min(count - 1, round(first + fps))
        record = dict(
            row,
            id="screen_" + row["id"],
            video_sha256=audit.digest(pathlib.Path(row["path"])),
            segment=dict(start_frame=first, end_frame=last, fps=fps),
            required_views=["orientation_screen_only"],
        )
        receipt = extract.extract_view(
            record, "primary", row["path"], destination=ROOT / "plank_screen"
        )
        with np.load(receipt["cache"]) as data:
            seq = data["image_xy"]
        choices = []
        for joints in [[11, 23, 25, 27], [12, 24, 26, 28]]:
            valid = np.isfinite(seq[:, joints]).all((1, 2))
            torso = np.linalg.norm(
                seq[:, joints[0]] - seq[:, joints[1]], axis=1
            )
            extent = np.abs(
                seq[:, joints[0], 0] - seq[:, joints[-1], 0]
            ) / np.maximum(torso, 1)
            choices.append(
                (
                    float(valid.mean()),
                    float(np.median(extent[valid])) if valid.any() else 0.0,
                )
            )
        observed, extent = max(
            choices, key=lambda x: (x[0] >= 0.8, x[1] if x[0] >= 0.8 else x[0])
        )
        results.append(
            dict(
                row,
                extraction=receipt,
                joint_chain_coverage=observed,
                horizontal_extent_torso_ratio=extent,
                side_candidate=observed >= 0.8 and extent >= 2,
                automatic_screen_only=True,
                directly_reviewed=False,
            )
        )
        audit.save_json(
            ROOT / "plank_screen.json", dict(records=results, completed=False)
        )
    ranked = sorted(
        results,
        key=lambda r: (
            r["side_candidate"],
            r["joint_chain_coverage"] >= 0.8,
            r["horizontal_extent_torso_ratio"],
        ),
        reverse=True,
    )
    prepared = []
    for row in ranked[:8]:
        receipt = external_review.contact_sheet(
            pathlib.Path(row["path"]),
            ROOT / "review" / (row["id"] + ".jpg"),
            0,
            None,
            24,
        )
        prepared.append(dict(row, preparation=receipt))
    report = dict(
        records=results,
        completed=True,
        prepared=prepared,
        side_candidate_count=sum(r["side_candidate"] for r in results),
        direct_visual_reviews=0,
        new_score_labels=0,
    )
    audit.save_json(ROOT / "plank_screen.json", report)
    return dict(
        screened=len(results),
        side_candidates=report["side_candidate_count"],
        prepared=len(prepared),
    )


if __name__ == "__main__":
    print(json.dumps(run()))
