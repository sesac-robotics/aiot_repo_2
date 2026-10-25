"""직접 검토한 MultiView 스쿼트만 별도 모델 후보로 비교한다.

기존 5종 모델·라벨·시험 분할은 변경하지 않는다. 두 시점의 같은 수행
구간에 대한 관측 정렬이며 촬영 장치의 하드웨어 동기화를 보증하지 않는다.
"""

import json
import pathlib
import time

import cv2
import joblib
import numpy as np

import audit
import extract
import recovery_features
import recovery_train
import three_items
from multiview_squat_roi import extract_masked
from multiview_squat_prepare import OUT
from recovery_prepare import ROOT
from resources import check_resources


def validate_alignment(
    decision: dict, side: np.ndarray, front: np.ndarray
) -> None:
    """승인된 관측 정렬과 실제 추출 프레임의 허용 오차를 검사한다.

    Args:
        decision: 직접 표본 대조 승인·시차·원본 프레임 구간.
        side: 측면 추출 프레임 번호, 정수 1차원 배열.
        front: 정면 추출 프레임 번호, 정수 1차원 배열.

    Returns:
        None: 같은 표본 수와 시차 대비 최대 1프레임 오차 확인 시 반환한다.
        15fps에서 10Hz 반올림에 생기는 오차이며 관절을 보간하지 않는다.

    Raises:
        ValueError: 미승인·불일치·과도한 프레임 오차 또는 표본 부족.
    """
    if not decision.get("paired_interval_approved_by_direct_samples"):
        raise ValueError("Direct paired interval review required")
    if side.ndim != 1 or front.shape != side.shape or len(side) < 5:
        raise ValueError("Equal nonempty temporal samples required")
    delta = front - side - decision["lag_frames"]
    if np.max(np.abs(delta)) > 1:
        raise ValueError("Frame offset differs from reviewed alignment")
    if np.any(np.diff(side) <= 0) or np.any(np.diff(front) <= 0):
        raise ValueError("Frames must strictly increase")


def extract_rows() -> list[dict]:
    """확정된 직접 초안 구간의 측면·정면 Heavy 관절 캐시를 준비한다.

    Returns:
        list[dict]: 직접 라벨·시점별 추출 계약·캐시 연결. 추출 자체는 직접
        영상 검토가 아니며 제공자 정상·오류 문자열을 사용하지 않는다.

    Raises:
        ValueError: 개발 역할·원본 해시 또는 추출 프레임 계약 불일치.
        RuntimeError: 온도·메모리 보호 기준 위반.
        OSError: 원본·초안·캐시 저장 실패.
    """
    labels = audit.read_json(OUT / "score_annotations.json")
    receipts = []
    for row in labels:
        if row["split"] not in ("train", "validation"):
            raise ValueError("Existing test sources cannot enter this trial")
        check_resources()
        primary = dict(row, required_views=["side", "synchronized_front"])
        front = dict(primary, segment=row["front_segment"])
        side_receipt = (
            extract_masked(primary)
            if row["id"] == "subject_002_squat_bad_paired_direct_rep0"
            else extract.extract_view(
                primary, "primary", row["video_path"], OUT / "heavy"
            )
        )
        front_receipt = extract.extract_view(
            front, "front", row["front_path"], OUT / "heavy"
        )
        with np.load(side_receipt["cache"]) as s, np.load(
            front_receipt["cache"]
        ) as f:
            validate_alignment(
                row["alignment"], s["frame_numbers"], f["frame_numbers"]
            )
        receipts.append(
            dict(id=row["id"], side=side_receipt, front=front_receipt)
        )
        audit.save_json(
            OUT / "extraction.json", dict(records=receipts, completed=False)
        )
    audit.save_json(
        OUT / "extraction.json", dict(records=receipts, completed=True)
    )
    return receipts


def load_new(receipts: list[dict]) -> list[dict]:
    """직접 등급과 점수 없는 관절 특징을 구도에 맞춰 연결한다.

    Args:
        receipts: extract_rows가 검증한 시점별 추출 결과.

    Returns:
        list[dict]: 기존 학습기의 항목 입력 형식. 누락 관절 항목은 보류한다.

    Raises:
        ValueError: 라벨·원본·캐시·구간·정렬 계약 불일치.
        OSError: 캐시 또는 원본 크기 조회 실패.
    """
    index = {r["id"]: r for r in receipts}
    review_path = OUT / "subject_tracking_review.json"
    tracking = (
        {
            (r["id"], r["view"]): r
            for r in audit.read_json(review_path)["records"]
        }
        if review_path.exists()
        else {}
    )
    result = []
    for row in audit.read_json(OUT / "score_annotations.json"):
        sequences, frames = [], []
        for view, path, segment, sha in [
            ("side", row["video_path"], row["segment"], row["video_sha256"]),
            (
                "front",
                row["front_path"],
                row["front_segment"],
                row["front_sha256"],
            ),
        ]:
            rec = index[row["id"]][view]
            if (
                rec["config"]["segment"] != segment
                or rec["config"]["source_sha256"] != sha
            ):
                raise ValueError("Cached input differs from direct label")
            if audit.digest(pathlib.Path(path)) != sha:
                raise ValueError("Source changed after direct review")
            with np.load(rec["cache"]) as data:
                if str(data["fingerprint"]) != rec["fingerprint"]:
                    raise ValueError("NPZ fingerprint mismatch")
                seq = data["image_xy"].copy()
                frames.append(data["frame_numbers"].copy())
            cap = cv2.VideoCapture(path)
            try:
                seq = three_items.mask_image_border(seq, cap.get(3), cap.get(4))
            finally:
                cap.release()
            sequences.append(seq)
        validate_alignment(row["alignment"], *frames)
        notes = three_items.describe("squats", *sequences)
        if not tracking:
            raise ValueError(
                "Direct temporal joint review is required before fitting"
            )
        for view in ("side", "front"):
            reviewed = tracking[(row["id"], view)]
            if reviewed["fingerprint"] != index[row["id"]][view]["fingerprint"]:
                raise ValueError("Tracking review does not match current cache")
            for i in reviewed.get("blocked_item_indices", []):
                notes[i]["eligible"] = False
                notes[i]["holds"].append(
                    "direct_sample_review_unreliable_target_geometry"
                )
        for i, note in enumerate(notes):
            note["bank"] = "front" if i == 3 else "side"
        assessment = row["assessment"]
        result.append(
            dict(
                id=row["id"],
                exercise="squats",
                person=row["person"],
                split=row["split"],
                source=row["video_sha256"],
                sources={row["video_sha256"], row["front_sha256"]},
                recording_groups=[row["recording_group"]],
                notes=notes,
                levels=assessment["criterion_levels"],
                weights=assessment["criterion_weights"],
                criterion_ids=assessment["criterion_ids"],
                reference_total_allowed=True,
                whole_attempt_observed=True,
            )
        )
    return result


def run() -> dict:
    """기존 기준 모델과 추가 학습 후보를 같은 개발 검증 행에서 비교한다.

    Returns:
        dict: 기존·추가 검증의 두 방식 오차, 관측 수, 분할 감사와 보존 해시.
        검증 자료를 모델 학습·경계 선택에 사용하지 않는다. 독립 최종 시험은
        아니며 비전문가 직접 표본 검토 초안과의 오차다.

    Raises:
        ValueError: 참여자·원본 누출 또는 보존 파일 변경.
        RuntimeError: 자원 보호 조건 위반.
        OSError: 모델·평가 결과 저장 실패.
    """
    protected = [ROOT / "comparison.json", ROOT / "fit_input_manifest.json"]
    protected += list((ROOT / "models").glob("*.joblib"))
    before = {str(p): audit.digest(p) for p in protected}
    extraction_started = time.perf_counter()
    receipts = extract_rows()
    extraction_elapsed = time.perf_counter() - extraction_started
    added = load_new(receipts)
    base, extra = recovery_features.load()
    original = [r for r in base + extra if r["exercise"] == "squats"]
    rows = original + added
    groups = recovery_train.component_groups(rows)
    training = [r for r in rows if r["split"] == "train"]
    model = recovery_train.fit(training)
    baseline = joblib.load(ROOT / "models/squats.joblib")
    joblib.dump(model, OUT / "squats_candidate.joblib")
    predictions = []
    added_ids = {r["id"] for r in added}
    for row in rows:
        if row["split"] != "validation":
            continue
        reference = (
            three_items.total(row["levels"], row["weights"])
            if row.get("reference_total_allowed", True)
            else None
        )
        predictions.append(
            dict(
                id=row["id"],
                person=row["person"],
                criterion_ids=row["criterion_ids"],
                levels=row["levels"],
                target_score=reference,
                subset=(
                    "added_multiview"
                    if row["id"] in added_ids
                    else "existing_development"
                ),
                baseline=recovery_train.predict(baseline, row),
                **recovery_train.predict(model, row),
                observation=[
                    dict(
                        item=n["item"],
                        eligible=n["eligible"],
                        holds=n.get("holds", []),
                    )
                    for n in row["notes"]
                ],
            )
        )
    comparisons = {}
    for subset in ("existing_development", "added_multiview"):
        selected = [p for p in predictions if p["subset"] == subset]
        old = [dict(p, **p["baseline"]) for p in selected]
        comparisons[subset] = dict(
            baseline=recovery_train.summarize(old),
            candidate=recovery_train.summarize(selected),
        )
    effective = [
        r
        for r in training
        if any(
            g is not None and n["eligible"]
            for g, n in zip(r["levels"], r["notes"])
        )
    ]
    if any(audit.digest(pathlib.Path(p)) != h for p, h in before.items()):
        raise ValueError("Preserved baseline changed")
    report = dict(
        completed=True,
        experiment="isolated_multiview_squat_development_trial",
        independent_final_test=False,
        expert_reference=False,
        promoted=False,
        source_comparison_sha256=before[str(ROOT / "comparison.json")],
        labels_sha256=audit.digest(OUT / "score_annotations.json"),
        manual_decisions_sha256=audit.digest(OUT / "manual_decisions.json"),
        subject_tracking_review_sha256=audit.digest(
            OUT / "subject_tracking_review.json"
        ),
        model_sha256=audit.digest(OUT / "squats_candidate.joblib"),
        code_sha256={
            name: audit.digest(pathlib.Path(__file__).with_name(name))
            for name in (
                "multiview_squat_trial.py",
                "multiview_squat_sync.py",
                "multiview_squat_roi.py",
                "recovery_features.py",
                "recovery_train.py",
                "three_items.py",
            )
        },
        preserved=before,
        training_effective_segments=len(effective),
        added_training_segments=sum(r["split"] == "train" for r in added),
        added_validation_segments=sum(
            r["split"] == "validation" for r in added
        ),
        added_training_eligible_item_labels=sum(
            n["eligible"] and g is not None
            for r in added
            if r["split"] == "train"
            for n, g in zip(r["notes"], r["levels"])
        ),
        added_validation_eligible_item_labels=sum(
            n["eligible"] and g is not None
            for r in added
            if r["split"] == "validation"
            for n, g in zip(r["notes"], r["levels"])
        ),
        added_four_item_observed_segments=sum(
            all(n["eligible"] for n in r["notes"]) for r in added
        ),
        training_metadata_groups=len({r["person"] for r in effective}),
        connected_groups=len(set(groups)),
        split_leakage=False,
        comparisons=comparisons,
        predictions=predictions,
        cached_receipt_extraction_seconds_sum=sum(
            c[view]["extract_seconds"]
            for c in receipts
            for view in ("side", "front")
        ),
        extraction_elapsed_seconds_this_call=extraction_elapsed,
        reused_view_receipts=sum(
            c[view]["reused"] for c in receipts for view in ("side", "front")
        ),
        timing_scope_ko="캐시 영수증의 과거 추출 소요 합과 이번 캐시 검증·재사용 경과 시간을 분리한다. 둘 다 서비스 실시간 지연 측정값이 아니다.",
        sampling_alignment_error_max_frames=1,
        camera_hardware_synchronization_verified=False,
        interpretation_ko="선택 구간의 같은 동작 사건을 표본 대조한 개발 실험. 전체 영상 동기화·실제 참여자 신원은 보증하지 않는다.",
    )
    audit.save_json(OUT / "comparison.json", report)
    audit.save_json(
        OUT / "feature_manifest.json",
        [
            dict(
                id=r["id"],
                split=r["split"],
                person=r["person"],
                sources=sorted(r["sources"]),
                levels=r["levels"],
                notes=[
                    dict(
                        item=n["item"],
                        bank=n["bank"],
                        eligible=n["eligible"],
                        feature=n["feature"] if n["eligible"] else None,
                        holds=n.get("holds", []),
                    )
                    for n in r["notes"]
                ],
            )
            for r in added
        ],
    )
    return {
        k: report[k]
        for k in (
            "completed",
            "training_effective_segments",
            "added_training_segments",
            "added_validation_segments",
            "comparisons",
        )
    }


if __name__ == "__main__":
    import sys

    if "extract-only" in sys.argv:
        print(
            json.dumps(dict(extracted=len(extract_rows())), ensure_ascii=False)
        )
    else:
        print(json.dumps(run(), ensure_ascii=False))
