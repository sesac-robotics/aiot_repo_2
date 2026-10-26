"""보유 자료의 같은 관측 항목을 구도별 특징 묶음으로 연결한다.

기존 공통 채점 코드를 수정하지 않는 후보다. 정면 투영 특징을 측면의
해부학적 각도와 같은 학습 묶음으로 취급하지 않는다.
"""

import pathlib

import cv2
import numpy as np

import audit
import geometry
import three_items
import three_pipeline
import ytw_partial_trial
import w_morphology
from plank_partial_item_trial import load_rows
from plank_item_observation import CHANNELS
from recovery_prepare import ROOT


def describe_ytw(seq: np.ndarray) -> list[dict]:
    """자동 구간 인식에서 보이는 Y·T·W와 전경 대리 특징을 반환한다.

    Args:
        seq: N×33×2 픽셀 관절. 누락은 NaN이며 보간하지 않는다.

    Returns:
        list[dict]: 네 항목의 관측·특징·규칙 등급. Y·전경의 규칙 총점은
        검증되지 않아 None이다. AI 입력에는 단계 정답·규칙 등급이 없다.

    Raises:
        ValueError: 좌표 또는 자동 구간 계약 오류.
    """
    phases = ytw_partial_trial.detect(seq)["phases"]
    channels = geometry.channels(seq)
    results = []
    for name, phase in [
        ("y_reach", "Y_attempt"),
        ("t_reach", "T_attempt"),
        ("w_shape", "W_attempt"),
    ]:
        window = phases.get(phase)
        feature = None
        grade = None
        if window:
            a, b = window["start_index"], window["end_index"]
            subset = seq[a : b + 1]
            if name == "w_shape":
                result = w_morphology.evaluate(seq, a, b)
                if result["grade"] is not None:
                    feature = [
                        result["arms"][s][k]
                        for s in ["left", "right"]
                        for k in [
                            "projected_bend_deg",
                            "hand_above_elbow_torso_ratio",
                            "elbow_up_upper_arm_ratio",
                        ]
                    ] + [result["bilateral_bend_difference_deg"]]
                    grade = result["grade"]
            elif (
                len(subset) >= 3
                and np.isfinite(subset[:, [11, 12, 13, 14, 15, 16, 23, 24]])
                .all((1, 2))
                .mean()
                >= 0.8
            ):
                elevation = np.minimum(
                    channels["left_elevation_deg"][a : b + 1],
                    channels["right_elevation_deg"][a : b + 1],
                )
                elbow = np.minimum(
                    channels["left_elbow_deg"][a : b + 1],
                    channels["right_elbow_deg"][a : b + 1],
                )
                feature = [
                    float(np.nanpercentile(x, q))
                    for x in [elevation, elbow]
                    for q in [10, 50, 90]
                ]
                if name == "t_reach":
                    error = abs(feature[1] - 90)
                    grade = 2 if error <= 15 else 1 if error <= 30 else 0
        results.append(
            dict(
                item=name,
                eligible=feature is not None,
                feature=feature,
                rule_grade=grade,
                measurement=None,
                higher_is_better=True,
                window=window,
            )
        )
    required = [11, 12, 23, 24]
    feature = None
    if (
        len(seq) >= 5
        and np.isfinite(seq[:, required]).all((1, 2)).mean() >= 0.8
    ):
        shoulder = (seq[:, 11] + seq[:, 12]) / 2
        hip = (seq[:, 23] + seq[:, 24]) / 2
        torso = np.linalg.norm(shoulder - hip, axis=1)
        ratio = np.linalg.norm(seq[:, 11] - seq[:, 12], axis=1) / np.maximum(
            torso, 1
        )
        tilt = np.degrees(
            np.arctan2(shoulder[:, 0] - hip[:, 0], hip[:, 1] - shoulder[:, 1])
        )
        feature = [
            float(np.nanpercentile(x, q))
            for x in [ratio, tilt]
            for q in [10, 50, 90]
        ]
    results.append(
        dict(
            item="hinge_and_trunk_hold",
            eligible=feature is not None,
            feature=feature,
            rule_grade=None,
            measurement=None,
            higher_is_better=True,
            proxy_only=True,
        )
    )
    return results


def load() -> tuple[list[dict], list[dict]]:
    """기존 3종·플랭크 입력과 보유 QEVD 추가 구도 입력을 분리 반환한다.

    Returns:
        tuple[list[dict],list[dict]]: 기존 입력, 추가 부분 입력. 각 행은
        관측 특징·직접 등급·원본 해시·참여자·개발 분할을 포함한다.
        test는 사용하지 않는다. 플랭크 지지 특징은 접촉 대리값이다.

    Raises:
        OSError: 라벨·캐시 접근 실패.
        ValueError: 원본·캐시 계약 또는 입력 구도 불일치.
    """
    base = []
    for row in three_pipeline.load():
        if (
            row["exercise"] == "squats"
            and row["manifest"].get("provenance", {}).get("dataset")
            != "REHAB24"
        ):
            continue
        for i, note in enumerate(row["notes"]):
            view = (
                "front"
                if row["exercise"] == "squats" and i == 3
                else (
                    "side"
                    if row["exercise"] != "shoulder"
                    else (
                        "front_oblique"
                        if "front_oblique"
                        in row["manifest"].get("required_input_views", [])
                        else "front"
                    )
                )
            )
            note["bank"] = view
        base.append(row)
    for row in load_rows(True):
        channels = geometry.channels(row["seq"])
        notes = []
        for index, observed in enumerate(row["observation"]["items"]):
            series = channels[observed["side"] + "_" + CHANNELS[index]]
            valid = series[np.isfinite(series)]
            feature = (
                [
                    float(np.percentile(valid, q))
                    for q in [5, 10, 25, 50, 75, 90, 95]
                ]
                if len(valid)
                else None
            )
            notes.append(
                dict(
                    item=observed["criterion_id"],
                    eligible=observed["eligible"] and feature is not None,
                    feature=feature,
                    measurement=observed["measured"],
                    higher_is_better=index in (0, 2),
                    rule_grade=observed["rule_grade"],
                    bank="side_video",
                )
            )
        base.append(
            dict(
                id=row["id"],
                exercise="plank",
                person=row["person"],
                source=row["source"],
                sources={row["source"]},
                split=row["split"],
                notes=notes,
                levels=row["grades"],
                weights=[40, 25, 20, 15],
                criterion_ids=[x["item"] for x in notes],
            )
        )
    plan = audit.read_json(ROOT / "input_plan.json")["records"]
    caches = {
        x["id"]: x["extraction"]
        for x in audit.read_json(ROOT / "extraction.json")["records"]
    }
    additional = []
    for row in plan:
        cached = caches[row["id"]]
        if (
            cached["config"]["source_sha256"] != row["video_sha256"]
            or cached["config"]["segment"] != row["segment"]
        ):
            raise ValueError("Input fingerprint mismatch")
        with np.load(cached["cache"]) as data:
            seq = data["image_xy"].copy()
        cap = cv2.VideoCapture(row["video_path"])
        width, height = cap.get(3), cap.get(4)
        cap.release()
        seq = three_items.mask_image_border(seq, width, height)
        e = row["exercise"]
        if e == "ytw":
            notes = describe_ytw(seq)
        else:
            notes = three_items.describe(e, seq, seq if e == "squats" else None)
            notes = three_items.apply_view_policy(
                e, notes, row["required_views"]
            )
            allowed = {
                "pushups": {1, 2},
                "squats": {0, 1, 3},
                "shoulder": {0, 1, 2},
            }[e]
            for i, note in enumerate(notes):
                note["eligible"] = note["eligible"] and i in allowed
        for i, note in enumerate(notes):
            note["bank"] = (
                "front" if e == "squats" and i == 3 else row["recovery_view"]
            )
        converted = dict(
            id=row["id"],
            exercise=e,
            person=row["person"],
            source=row["video_sha256"],
            sources={row["video_sha256"]},
            split=row["split"],
            notes=notes,
            levels=row["criterion_levels"],
            weights=row["criterion_weights"],
            criterion_ids=row["criterion_ids"],
        )
        # 기존 YTW 입력도 이번 후보에서는 자동 구간 기반의 새 특징으로 함께 학습한다.
        (
            base
            if e == "ytw" and not row["id"].endswith("_existing_review")
            else additional
        ).append(converted)
    return base, additional
