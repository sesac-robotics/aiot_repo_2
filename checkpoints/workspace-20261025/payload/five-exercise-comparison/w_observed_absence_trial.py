"""충분히 관측된 T 이후 W 없이 팔을 내리는 실패 수행을 별도 검증한다.

인식 실패 자체를 0점으로 바꾸지 않는다. 양팔·몸통이 실제 관측되고
T 뒤 W 손·팔 굽힘 근거가 없으며 팔 내림 유지가 확인된 경우만 후보를 낸다.
"""

import json
import pathlib

import cv2
import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

import audit
import geometry
import three_items
import w_morphology
import ytw_phases
from allfive_round1 import OUT
from allfive_round1_features import phase_matches
from multiview_squat_batch2_train import mask_reviewed_frames
from recovery_prepare import ROOT
from recovery_train import component_groups
from resources import check_resources
from ytw_projected_rule_trial import fit, predict

DEST = OUT / "w_observed_absence_trial"


def detect(sequence: np.ndarray) -> dict | None:
    """점수 없이 관측된 T→팔 내림과 중간 W 근거의 부재를 확인한다.

    Args:
        sequence: 10Hz N×33×2 원본 픽셀 좌표. 누락은 NaN이다.

    Returns:
        dict | None: T·팔 내림 원본 표본 인덱스와 실제 W 형태 측정 또는
        근거 부족. 후반 관측 80%·6표본·팔 내림 3표본 이상을 요구한다.
        W 손 높이·팔 굽힘이 3표본 연속 보이면 후보를 만들지 않는다.

    Raises:
        ValueError: 좌표·빈 입력 또는 기하 형상 계약 위반.
    """
    seq = np.asarray(sequence, dtype=float)
    if seq.ndim != 3 or seq.shape[1:] != (33, 2) or not len(seq):
        raise ValueError("Expected observed N x 33 x 2")
    if len(seq) < 12:
        return None
    required = [11, 12, 13, 14, 15, 16, 23, 24]
    valid = np.isfinite(seq[:, required]).all((1, 2))
    shoulder = (seq[:, 11] + seq[:, 12]) / 2
    hip = (seq[:, 23] + seq[:, 24]) / 2
    torso = np.linalg.norm(shoulder - hip, axis=1)
    up = np.divide(
        shoulder - hip,
        torso[:, None],
        out=np.full_like(shoulder, np.nan),
        where=torso[:, None] > 1,
    )
    lateral = np.stack([-up[:, 1], up[:, 0]], axis=1)
    wrist = seq[:, [15, 16]] - shoulder[:, None]
    width = np.mean(abs(np.sum(wrist * lateral[:, None], axis=2)), axis=1)
    width = np.where(valid, width, np.nan)
    smooth = np.full(len(seq), np.nan)
    for i in range(1, len(seq) - 1):
        if valid[i - 1 : i + 2].all():
            smooth[i] = np.median(width[i - 1 : i + 2])
    if np.isfinite(smooth).sum() < 9:
        return None
    peak = float(np.nanpercentile(smooth, 90))
    if peak < 1:
        return None
    wide = ytw_phases._runs(valid & (smooth >= peak * 0.85), 3)
    if not wide:
        return None
    a, b = max(wide, key=lambda r: (r[1] - r[0], -r[0]))
    post = np.arange(len(seq)) > b
    if post.sum() < 6 or valid[post].mean() < 0.8:
        return None
    height = []
    bend = []
    for s, e, w in ((11, 13, 15), (12, 14, 16)):
        height.append(
            np.sum((seq[:, w] - seq[:, e]) * up, axis=1) / np.maximum(torso, 1)
        )
        bend.append(180 - geometry.angle_deg(seq[:, s], seq[:, e], seq[:, w]))
    W_evidence = (
        post
        & valid
        & (np.minimum(*height) >= -0.05)
        & (np.minimum(*bend) >= 15)
    )
    if ytw_phases._runs(W_evidence, 3):
        return None
    hand_down = np.mean(
        np.sum(wrist * (-up[:, None]), axis=2), axis=1
    ) / np.maximum(torso, 1)
    low = (
        post
        & valid
        & (smooth <= peak * 0.55)
        & (hand_down >= 0.6)
        & (np.maximum(*height) < -0.1)
    )
    runs = ytw_phases._runs(low, 3)
    if not runs:
        return None
    start, end = runs[0]
    if start - b > 10:
        return None
    morphology = w_morphology.evaluate(seq, start, end)
    if morphology["grade"] is None:
        return None
    feature = [
        morphology["arms"][s][k]
        for s in ("left", "right")
        for k in (
            "projected_bend_deg",
            "hand_above_elbow_torso_ratio",
            "elbow_up_upper_arm_ratio",
        )
    ] + [morphology["bilateral_bend_difference_deg"]]
    return dict(
        window=dict(
            start_index=start, end_index=end, anchor_index=(start + end) // 2
        ),
        T_window=[a, b],
        feature=feature,
        measured_shape_grade=morphology["grade"],
        post_observed_fraction=float(valid[post].mean()),
        actual_contact_not_inferred=True,
    )


def run() -> dict:
    """직접 라벨을 보존하고 관측 근거 후보의 W 오판·그룹 검증을 비교한다.

    Returns:
        dict: 새로운 관측 후보·직접 시점 대조·같은 자료의 총점·항목 오차.
        추가 규칙의 숫자는 검증 라벨로 선택하지 않으며 자동 승격하지 않는다.

    Raises:
        ValueError: 원본·직접 구간·캐시·역할·기존 결과 지문 불일치.
        OSError: 원본·캐시·감사 자료 저장 실패.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    DEST.mkdir(exist_ok=True)
    rows = [
        r
        for r in audit.read_json(OUT / "candidate_input_manifest.json")
        if r["exercise"] == "ytw"
    ]
    annotations = {
        r["id"]: r for r in audit.read_json(ROOT / "input_plan.json")["records"]
    }
    annotations.update(
        {r["id"]: r for r in audit.read_json(OUT / "score_annotations.json")}
    )
    caches = {
        r["id"]: r["extraction"]
        for r in audit.read_json(ROOT / "extraction.json")["records"]
    }
    caches.update(
        {
            r["id"]: r["extraction"]
            for r in audit.read_json(OUT / "extraction.json")["records"]
        }
    )
    manual = audit.read_json(OUT / "manual_decisions.json")[
        "phase_windows_native_frames"
    ]
    tracking = {
        r["id"]: r
        for r in audit.read_json(OUT / "subject_tracking_review.json")[
            "records"
        ]
    }
    records = []
    for row in rows:
        if row["levels"][2] is None:
            continue
        check_resources()
        rec = caches[row["id"]]
        ann = annotations[row["id"]]
        with np.load(rec["cache"]) as data:
            seq = mask_reviewed_frames(
                data["image_xy"],
                data["frame_numbers"],
                tracking.get(row["id"], {}).get(
                    "unreliable_frame_intervals", []
                ),
            )
            frames = data["frame_numbers"]
        cap = cv2.VideoCapture(ann["video_path"])
        seq = three_items.mask_image_border(seq, cap.get(3), cap.get(4))
        cap.release()
        result = detect(seq)
        if result is None:
            continue
        source_id = ann.get("source_id")
        matched = (
            phase_matches(
                result["window"], frames, manual[source_id]["w_shape"]
            )
            if source_id in manual
            else None
        )
        entry = dict(
            id=row["id"],
            split=row["split"],
            reference_W=row["levels"][2],
            before_eligible=row["items"][2]["eligible"],
            direct_window_matches=matched,
            **result
        )
        records.append(entry)
        # 이번에 직접 시점·관절을 대조한 신규 구간만 학습 입력을 보강한다.
        # 기존 구간의 불일치는 자동으로 재라벨·재학습하지 않고 진단에 남긴다.
        if (
            matched
            and row["id"] in tracking
            and tracking[row["id"]]["direct_temporal_sample_review"]
        ):
            row["items"][2].update(eligible=True, feature=result["feature"])
            entry["accepted_as_feature_not_label"] = True
        else:
            entry["accepted_as_feature_not_label"] = False
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    groups = component_groups(train)
    oof = []
    for fitting, held in GroupKFold(3).split(train, groups=groups):
        heads = fit([train[int(i)] for i in fitting])
        oof.extend(predict(heads, train[int(i)]) for i in held)
    heads = fit(train)
    joblib.dump(heads, DEST / "model.joblib")
    fixed = [predict(heads, r) for r in val]
    result = dict(
        completed=True,
        observed_absence_candidates=records,
        accepted_new_feature_rows=sum(
            r["accepted_as_feature_not_label"] for r in records
        ),
        candidate_input_sha256=audit.digest(
            OUT / "candidate_input_manifest.json"
        ),
        group_cv=recovery_train_summary(oof),
        fixed_development=recovery_train_summary(fixed),
        group_cv_predictions=oof,
        fixed_predictions=fixed,
        model_sha256=audit.digest(DEST / "model.joblib"),
        code_sha256=audit.digest(pathlib.Path(__file__)),
        labels_modified=False,
        validation_used_for_numeric_selection=False,
        old_diagnostic_rows_not_automatically_adopted=True,
        promoted=False,
        independent_final_test=False,
    )
    audit.save_json(DEST / "comparison.json", result)
    return dict(
        accepted=result["accepted_new_feature_rows"],
        fixed=result["fixed_development"],
    )


def recovery_train_summary(rows: list[dict]) -> dict:
    """동일 평가 행의 항목·총점 오차를 기존 집계 구현으로 반환한다.

    Args:
        rows: 독립 직접 등급과 두 방식 예측 목록.

    Returns:
        dict: 동일 관측 항목·총점 교집합 오차와 판정 가능 수.
    """
    from recovery_train import summarize

    return summarize(rows)


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
