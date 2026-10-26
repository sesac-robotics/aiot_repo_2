"""QEVD의 실제 분류 문자열로 후보를 균형 선정하고 관측 구도만 자동 선별한다.

중간 1.2초 Heavy 관절은 구도 후보 선정 전용이다. 직접 검토·점수 라벨이
아니며 새 최종 시험 원본은 이 실행에서 추출하거나 보지 않고 사전 예약만 한다.
"""

import hashlib
import pathlib

import cv2
import numpy as np

import audit
import external_review
import extract
import three_protocol
from resources import check_resources


def run() -> dict:
    """학습 48명·검증 24명의 새 원본을 자동 선별하고 상위 구도를 준비한다.

    Returns:
        dict: 자동 추출 수·측면 후보 수·검토판과 12명 이하 최종 예약 원본.
        평가 모델 예측과 직접 점수는 생성하지 않는다.

    Raises:
        OSError: 원본·캐시·결과 접근 실패.
        ValueError: 원본 메타데이터·관절 추출 계약 오류.
        RuntimeError: 전원·온도·메모리 보호 조건 위반.
    """
    root = three_protocol.OUT
    pool = audit.read_json(
        audit.BASE / "output/overnight-20261020/score_candidates.json"
    )
    known = {r["person"] for r in audit.read_json(audit.OUT / "labels.json")}
    known.update(
        r["person"]
        for r in audit.read_json(root / "candidate_preparation.json")["records"]
    )
    selected, reserved = [], []
    terms = [
        "90 degrees",
        "deep",
        "shallow",
        "elevated hips",
        "legs on the floor",
        "early stopping",
    ]
    for split, limit in [("train", 48), ("validation", 24), ("test", 12)]:
        eligible = [
            r
            for r in pool
            if r["exercise"] == "pushups"
            and r["id"].startswith("qevd_")
            and r["split"] == split
            and r["person"] not in known
        ]
        eligible.sort(
            key=lambda r: hashlib.sha256(
                ("three-view-screen-20261022" + r["id"]).encode()
            ).hexdigest()
        )
        people = set()
        for index in range(limit):
            candidates = [r for r in eligible if r["person"] not in people]
            row = next(
                (
                    r
                    for r in candidates
                    if any(
                        terms[index % len(terms)] in text
                        for text in r["source_labels"]
                    )
                ),
                candidates[0] if candidates else None,
            )
            if row is None:
                break
            people.add(row["person"])
            (reserved if split == "test" else selected).append(row)
    reservation = root / "pushup_prospective_reservation.json"
    if not reservation.exists():
        audit.save_json(
            reservation,
            dict(
                records=reserved,
                raw_frames_or_scores_viewed=False,
                selection="fixed_id_hash_and_provider_terms_only_no_score_conversion",
                previous_labelled_participants_excluded=True,
                prior_nonlabel_candidate_screening_audit_still_required=True,
                independent_final_qualification_not_yet_confirmed=True,
            ),
        )
    results = []
    destination = root / "view_screen"
    for candidate in selected:
        check_resources()
        path = pathlib.Path(candidate["path"])
        cap = cv2.VideoCapture(
            str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        try:
            fps, count = cap.get(5), int(cap.get(7))
        finally:
            cap.release()
        start = max(0, int(count / 2 - 0.6 * fps))
        row = dict(
            id="view_" + candidate["id"],
            segment=dict(
                start_frame=start,
                end_frame=min(count - 1, start + round(1.2 * fps)),
                fps=fps,
            ),
            video_sha256=audit.digest(path),
            required_views=["view_screen_only_no_score"],
        )
        receipt = extract.extract_view(
            row, "primary", str(path), destination=destination
        )
        with np.load(receipt["cache"]) as data:
            seq = data["image_xy"]
        chains = []
        for joints in [[11, 15, 23, 25, 27, 31], [12, 16, 24, 26, 28, 32]]:
            visible = np.isfinite(seq[:, joints]).all(axis=(1, 2))
            torso = np.linalg.norm(
                seq[:, joints[0]] - seq[:, joints[2]], axis=1
            )
            horizontal = np.abs(seq[:, joints[0], 0] - seq[:, joints[4], 0])
            ratios = horizontal[visible] / np.maximum(torso[visible], 1)
            chains.append(
                (
                    float(visible.mean()),
                    float(np.median(ratios)) if len(ratios) else 0,
                )
            )
        observed, extent = max(chains)
        results.append(
            dict(
                candidate,
                video_path=str(path),
                video_sha256=row["video_sha256"],
                simultaneous_chain_fraction=observed,
                horizontal_extent_torso_ratio=extent,
                side_candidate=observed >= 0.8 and extent >= 1.5,
                automatic_view_only=True,
                directly_reviewed=False,
            )
        )
        audit.save_json(
            root / "pushup_view_screen_checkpoint.json",
            dict(records=results, completed=False),
        )
    review = root / "review/pushup_view_screen"
    review.mkdir(parents=True, exist_ok=True)
    sheets = []
    for split, limit in [("train", 12), ("validation", 8)]:
        candidates = [
            r for r in results if r["split"] == split and r["side_candidate"]
        ]
        candidates.sort(
            key=lambda r: (-r["horizontal_extent_torso_ratio"], r["id"])
        )
        for row in candidates[:limit]:
            row["sheet"] = external_review.contact_sheet(
                pathlib.Path(row["video_path"]),
                review / (row["id"] + ".jpg"),
                0,
                None,
                36,
            )
            sheets.append(row)
    report = dict(
        records=results,
        prepared_for_direct_review=sheets,
        automatically_screened=len(results),
        directly_reviewed=0,
        score_labels_created=0,
        final_reserved=len(reserved),
        completed=True,
    )
    audit.save_json(root / "pushup_view_screen.json", report)
    return report


if __name__ == "__main__":
    print(run()["automatically_screened"])
