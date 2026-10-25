"""5종 별도 후보의 관측 특징과 직접 추적 승인·구도 계약을 연결한다."""

import copy
import pathlib

import cv2
import numpy as np

import audit
import geometry
import plank_item_observation
import recovery_features
import recovery_w_phase
import three_items
import upper_support_event_model
import w_morphology
from allfive_round1 import OUT
from multiview_squat_batch2 import OUT as SQUAT_BATCH2, PREVIOUS
from multiview_squat_batch2_train import load_batch, mask_reviewed_frames
from recovery_prepare import ROOT


def phase_matches(
    window: dict | None, frames: np.ndarray, interval: list[int]
) -> bool:
    """자동 단계의 중심·과반 표본이 직접 표시한 시도 구간에 있는지 검사한다.

    Args:
        window: 자동 인식기의 시작·끝·중심 인덱스 또는 None.
        frames: 추출 표본의 원본 프레임 번호.
        interval: 직접 표시한 해당 단계 시도의 포함 시작·끝 프레임.

    Returns:
        bool: 중심과 과반 표본이 구간 안이면 True. 직접 시점을 자동 단계
        선택에 넘기지 않으며 불일치 라벨·입력의 결합만 차단한다.

    Raises:
        ValueError: 자동 단계 인덱스 또는 직접 구간이 유효하지 않은 경우.
    """
    if window is None:
        return False
    a, b = window["start_index"], window["end_index"]
    anchor = window.get("anchor_index", (a + b) // 2)
    first, last = interval
    if not 0 <= a <= anchor <= b < len(frames) or first > last:
        raise ValueError("Invalid stage/frame contract")
    chosen = frames[a : b + 1]
    return bool(
        first <= frames[anchor] <= last
        and ((chosen >= first) & (chosen <= last)).mean() >= 0.5
    )


def pitch_features(sequence: np.ndarray) -> list[float] | None:
    """한쪽 발뒤꿈치–발끝의 화면상 기울기를 도 단위로 요약한다.

    Args:
        sequence: N×33×2 원본 픽셀 좌표. 누락은 NaN이다.

    Returns:
        list[float] | None: 발끝보다 뒤꿈치가 높을 때 양수인 각도의 중앙값·
        90백분위. 80% 동시 관측·5개 표본·발 길이 3픽셀 이상이 필요하다.
        지면 접촉이나 실제 발 각도를 인증하지 않는 측면 투영 특징이다.

    Raises:
        ValueError: 좌표 형상이 올바르지 않은 경우.
    """
    seq = np.asarray(sequence, dtype=float)
    if seq.ndim != 3 or seq.shape[1:] != (33, 2):
        raise ValueError("Expected N x 33 x 2 pixels")
    options = []
    for heel, toe in ((29, 31), (30, 32)):
        delta = seq[:, toe] - seq[:, heel]
        valid = np.isfinite(delta).all(1) & (np.linalg.norm(delta, axis=1) >= 3)
        if len(seq) >= 5 and valid.mean() >= 0.8:
            angles = np.degrees(
                np.arctan2(delta[valid, 1], abs(delta[valid, 0]))
            )
            options.append(
                (float(valid.mean()), np.percentile(angles, [50, 90]).tolist())
            )
    return max(options, key=lambda x: x[0])[1] if options else None


def refined_w(sequence: np.ndarray) -> dict | None:
    """점수·제공자 정보 없이 자동 W 시점과 실제 관측 특징을 구한다.

    Args:
        sequence: N×33×2 관측 픽셀 좌표.

    Returns:
        dict | None: 자동 시점·7개 화면 기하·명시 형태 규칙 또는 관측 불가.
        기존 개발 오판을 계기로 설계한 별도 후보이며 독립 검증은 아니다.

    Raises:
        ValueError: 좌표·표본·관측 형상 계약 위반.
    """
    detected = recovery_w_phase.detect(sequence)
    window = detected["phases"].get("W_attempt")
    if not window:
        return None
    result = w_morphology.evaluate(
        sequence, window["start_index"], window["end_index"]
    )
    if result["grade"] is None:
        return None
    feature = [
        result["arms"][side][key]
        for side in ("left", "right")
        for key in (
            "projected_bend_deg",
            "hand_above_elbow_torso_ratio",
            "elbow_up_upper_arm_ratio",
        )
    ] + [result["bilateral_bend_difference_deg"]]
    return dict(
        feature=feature,
        window=window,
        rule_grade=result["grade"],
        w_refined=detected.get("w_refined", False),
    )


def new_rows() -> tuple[list[dict], dict]:
    """새 직접 초안과 승인 관절의 교집합만 항목별 입력으로 반환한다.

    Returns:
        tuple: 새 점수 입력 행과 ID별 원본 관측 배열. 사건 미관측을 정상
        등급으로 바꾸지 않으며 정면 플랭크를 측면 총점 입력에 섞지 않는다.

    Raises:
        ValueError: 원본·캐시 지문·직접 검토 승인·구도 계약 불일치.
        OSError: 주석·원본·캐시 읽기 실패.
    """
    annotations = audit.read_json(OUT / "score_annotations.json")
    receipts = {
        r["id"]: r["extraction"]
        for r in audit.read_json(OUT / "extraction.json")["records"]
    }
    reviews = {
        r["id"]: r
        for r in audit.read_json(OUT / "subject_tracking_review.json")[
            "records"
        ]
    }
    phase_windows = audit.read_json(OUT / "manual_decisions.json")[
        "phase_windows_native_frames"
    ]
    sequences = {}
    rows = []
    for ann in annotations:
        rec = receipts[ann["id"]]
        review = reviews[ann["id"]]
        if (
            not review["direct_temporal_sample_review"]
            or review["fingerprint"] != rec["fingerprint"]
        ):
            raise ValueError("Direct tracking approval differs")
        if (
            rec["config"]["segment"] != ann["segment"]
            or audit.digest(pathlib.Path(ann["video_path"]))
            != ann["video_sha256"]
        ):
            raise ValueError("Source contract differs")
        with np.load(rec["cache"]) as data:
            if str(data["fingerprint"]) != rec["fingerprint"]:
                raise ValueError("Cache changed")
            seq = mask_reviewed_frames(
                data["image_xy"],
                data["frame_numbers"],
                review.get("unreliable_frame_intervals", []),
            )
            frames = data["frame_numbers"].copy()
        cap = cv2.VideoCapture(ann["video_path"])
        width, height = cap.get(3), cap.get(4)
        cap.release()
        seq = three_items.mask_image_border(seq, width, height)
        e = ann["exercise"]
        assessment = ann["assessment"]
        levels = assessment["criterion_levels"]
        side = "side" in ann["required_input_views"]
        if e == "ytw":
            notes = recovery_features.describe_ytw(seq)
        elif e == "plank" and side:
            observation = plank_item_observation.assess(
                seq,
                (ann["segment"]["end_frame"] - ann["segment"]["start_frame"])
                / ann["segment"]["fps"],
                [True] * 4,
            )
            channels = geometry.channels(seq)
            notes = []
            for i, n in enumerate(observation["items"]):
                series = channels[
                    n["side"] + "_" + plank_item_observation.CHANNELS[i]
                ]
                valid = series[np.isfinite(series)]
                notes.append(
                    dict(
                        item=n["criterion_id"],
                        eligible=n["eligible"] and len(valid) > 0,
                        feature=(
                            np.percentile(
                                valid, [5, 10, 25, 50, 75, 90, 95]
                            ).tolist()
                            if len(valid)
                            else None
                        ),
                        measurement=n["measured"],
                        higher_is_better=i in (0, 2),
                        rule_grade=n["rule_grade"],
                        bank="side_video",
                        holds=n["reasons"],
                    )
                )
        elif e == "plank":
            x, event = upper_support_event_model.event_features(seq)
            notes = [
                dict(
                    item=item,
                    eligible=i == 3 and x is not None and np.isfinite(x).all(),
                    feature=x.tolist() if i == 3 and x is not None else None,
                    measurement=None,
                    higher_is_better=False,
                    rule_grade=None,
                    bank="front_visible_interruption",
                    holds=["not_a_full_plank_observation"],
                )
                for i, item in enumerate(assessment["criterion_ids"])
            ]
        else:
            notes = three_items.describe(e, seq, seq if e == "squats" else None)
            allowed = {
                "pushups": {1, 2, 3} if levels[3] is not None else {1, 2},
                "squats": {0, 1, 3},
                "shoulder": {0, 1, 2, 3},
            }[e]
            for i, n in enumerate(notes):
                n["eligible"] = n["eligible"] and i in allowed
        for i, n in enumerate(notes):
            n.setdefault("holds", [])
            n.setdefault("bank", "side" if side else "front")
            if i in review.get("blocked_item_indices", []):
                n["eligible"] = False
                n["holds"].append("direct_tracking_review_unreliable_geometry")
            # 명시적으로 판정 불가인 새 구도는 모델의 확신으로 채우지 않는다.
            if levels[i] is None:
                n["eligible"] = False
                n["holds"].append(
                    "direct_item_not_observable_or_grade_boundary_unresolved"
                )
            if e == "ytw" and i < 3 and levels[i] is not None:
                interval = phase_windows[ann["source_id"]][n["item"]]
                if not phase_matches(n.get("window"), frames, interval):
                    n["eligible"] = False
                    n["holds"].append(
                        "automatic_phase_does_not_match_direct_label_interval"
                    )
        rows.append(
            dict(
                id=ann["id"],
                exercise=e,
                person=ann["person"],
                source=ann["video_sha256"],
                sources={ann["video_sha256"]},
                split=ann["split"],
                recording_groups=[],
                notes=notes,
                levels=levels,
                weights=assessment["criterion_weights"],
                criterion_ids=assessment["criterion_ids"],
                reference_total_allowed=ann[
                    "eligible_for_total_score_training"
                ],
                whole_attempt_observed=ann["attempt_complete"],
                manual_phase_windows=phase_windows.get(ann["source_id"], {}),
                phase_review_permitted=[
                    g is not None
                    and i not in review.get("blocked_item_indices", [])
                    for i, g in enumerate(levels)
                ],
                frame_numbers=frames,
            )
        )
        sequences[ann["id"]] = seq
    return rows, sequences


def load() -> tuple[list[dict], list[dict], dict]:
    """보존 기준 입력과 새 특징 후보를 같은 역할·원본으로 준비한다.

    Returns:
        tuple: 기준 특징 행, 후보 특징 행, 자동 특징 변경 진단. 라벨 값은
        특징 계산에 사용하지 않으며 모델·평가 역할과 기존 캐시는 보존한다.

    Raises:
        ValueError: 동일 원본·참여자·캐시·기존 입력 지문 불일치.
        OSError: 기존·추가 주석·캐시 접근 실패.
    """
    base, extra = recovery_features.load()
    rows = base + extra
    rows.extend(load_batch(PREVIOUS))
    rows.extend(load_batch(SQUAT_BATCH2))
    added, sequences = new_rows()
    rows.extend(added)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate score IDs")
    # 기존 raw 캐시를 바꾸지 않고 이번 특징 계산에만 사용한다.
    for r in rows:
        if "primary" in r:
            sequences.setdefault(r["id"], r["primary"])
    maps = []
    source_paths = {
        r["id"]: r["video_path"]
        for r in audit.read_json(ROOT / "input_plan.json")["records"]
    }
    for folder in (PREVIOUS, SQUAT_BATCH2):
        source_paths.update(
            {
                r["id"]: r["video_path"]
                for r in audit.read_json(folder / "score_annotations.json")
            }
        )
    for folder in (ROOT, PREVIOUS, SQUAT_BATCH2):
        records = audit.read_json(folder / "extraction.json")["records"]
        tracking = (
            {
                r["id"]: r
                for r in audit.read_json(
                    folder / "subject_tracking_review.json"
                )["records"]
                if r.get("view") == "side"
            }
            if (folder / "subject_tracking_review.json").exists()
            else {}
        )
        for r in records:
            rec = r.get("extraction", r.get("side"))
            if rec is None:
                continue
            maps.append((r["id"], rec, tracking.get(r["id"], {})))
    needed = {
        r["id"]
        for r in rows
        if r["exercise"] in ("ytw", "squats") and r["id"] not in sequences
    }
    for identifier, rec, review in maps:
        if identifier not in needed:
            continue
        with np.load(rec["cache"]) as data:
            if str(data["fingerprint"]) != rec["fingerprint"]:
                raise ValueError("Feature cache changed")
            sequences[identifier] = mask_reviewed_frames(
                data["image_xy"],
                data["frame_numbers"],
                review.get("unreliable_frame_intervals", []),
            )
        cap = cv2.VideoCapture(source_paths[identifier])
        try:
            sequences[identifier] = three_items.mask_image_border(
                sequences[identifier], cap.get(3), cap.get(4)
            )
        finally:
            cap.release()
    candidate = copy.deepcopy(rows)
    changes = []
    for row in candidate:
        seq = sequences.get(row["id"])
        if seq is None:
            continue
        if (
            row["exercise"] == "squats"
            and row["notes"][2]["bank"] == "side"
            and row["notes"][2]["eligible"]
        ):
            pitch = pitch_features(seq)
            if pitch is not None:
                n = row["notes"][2]
                n["feature"] = [*n["feature"][:-1], *pitch, n["feature"][-1]]
                n["bank"] = "side_pitch_v1"
                n["pitch_feature_indices"] = [8, 9]
                n["pitch_units"] = "degrees_image_projection"
                changes.append(
                    dict(
                        id=row["id"], kind="side_heel_toe_pitch", feature=pitch
                    )
                )
        if row["exercise"] == "ytw" and (
            row["notes"][2]["eligible"]
            or (
                row.get("manual_phase_windows")
                and row["phase_review_permitted"][2]
            )
        ):
            refined = refined_w(seq)
            if (
                refined
                and row.get("manual_phase_windows")
                and not phase_matches(
                    refined["window"],
                    row["frame_numbers"],
                    row["manual_phase_windows"]["w_shape"],
                )
            ):
                refined = None
            if refined:
                row["notes"][2].update(refined)
                row["notes"][2]["eligible"] = True
                changes.append(
                    dict(
                        id=row["id"],
                        kind="automatic_W_phase_candidate",
                        window=refined["window"],
                        refined=refined["w_refined"],
                    )
                )
    return (
        rows,
        candidate,
        dict(records=changes, features_are_automatic_not_direct_labels=True),
    )
