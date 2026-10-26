"""보존된 관절·추적 마스크로 보유 자료의 지지·발 특징만 개선한다."""

import collections
import copy

import cv2
import numpy as np

import audit
import held_geometry
import recovery_train
import three_items
import three_training
from allfive_round1_train import fit as base_fit
from multiview_squat_batch2 import OUT as BATCH2, PREVIOUS
from multiview_squat_batch2_train import mask_reviewed_frames
from plank_partial_item_trial import load_rows
from recovery_prepare import ROOT


def update_note(
    row: dict, sequence: np.ndarray, allow_partial_top: bool = False
) -> dict | None:
    """이미 승인된 항목의 필요 관측을 추가 검사하여 입력 사본을 갱신한다.

    Args:
        row: notes·exercise가 있는 관측 입력. 직접 등급을 읽지 않는다.
        sequence: N×33×2 픽셀 관절. 이전 직접 추적 마스크가 적용된 좌표다.
        allow_partial_top: 별도 v2 실험에서 부분 복귀의 발 관측을 허용한다.

    Returns:
        dict | None: 변경 진단 또는 대상 항목이 아닌 경우 None. 함수는
        해당 행의 항목 특징을 수정하므로 호출자가 행 사본을 전달해야 한다.

    Raises:
        ValueError: 좌표·기하 특징 형상 계약 위반.
    """
    note = row["notes"][2]
    if not note["eligible"]:
        return None
    if row["exercise"] == "plank" and note["bank"] == "side_video":
        result = held_geometry.plank_clearance(sequence)
        note.update(
            bank="side_support_clearance_v1",
            feature=result["feature"],
            measurement=result["measurement"],
            eligible=result["eligible"],
            rule_grade=held_geometry.plank_support_grade(result["measurement"]),
            required_joint_indices=[11, 13, 23, 25, 27, 31],
            measurement_units=["degrees", "torso_ratio", "torso_ratio"],
            contact_is_proxy=True,
        )
    elif row["exercise"] == "squats" and note["bank"] == "side":
        result = held_geometry.squat_foot_change(sequence, allow_partial_top)
        note.update(
            bank=(
                "side_foot_partial_top_v2"
                if allow_partial_top
                else "side_foot_change_v1"
            ),
            eligible=result["eligible"],
            feature=(
                note["feature"][:-1] + result["feature"]
                if result["eligible"]
                else None
            ),
            heel_units="standing_to_bottom_height_change_over_fixed_foot_length",
            foot_phase_measurement_is_automatic=True,
        )
    else:
        return None
    note.setdefault("holds", []).extend(result["holds"])
    return dict(
        id=row["id"],
        bank=note["bank"],
        eligible=note["eligible"],
        measurement=note["measurement"],
        feature=note["feature"],
        holds=result["holds"],
    )


def update_existing(
    rows: list[dict], allow_partial_top: bool = False
) -> tuple[list[dict], dict]:
    """기존 승인·분할을 보존하며 플랭크와 스쿼트의 관절 특징을 갱신한다.

    Args:
        rows: 기존 오프라인 학습·개발 입력. 입력 행은 변경하지 않는다.
        allow_partial_top: True이면 별도 v2 발 관측을 사용한다. 기본은 기존 v1이다.

    Returns:
        tuple: 변경 사본과 자동 특징 변경 진단. 새로운 직접 검토·등급이나
        독립 시험 자료를 만들지 않는다. 관절 누락은 보간하지 않는다.

    Raises:
        ValueError: 캐시 지문·추적 승인 마스크 불일치.
        OSError: 기존 캐시·주석·원본 접근 실패.
    """
    result = copy.deepcopy(rows)
    sequences = {r["id"]: r["seq"] for r in load_rows(True)}
    for r in rows:
        if "primary" in r and r["primary"] is not None:
            sequences.setdefault(r["id"], r["primary"])
    for folder in (PREVIOUS, BATCH2):
        annotations = {
            r["id"]: r
            for r in audit.read_json(folder / "score_annotations.json")
        }
        tracking = {
            (r["id"], r["view"]): r
            for r in audit.read_json(folder / "subject_tracking_review.json")[
                "records"
            ]
        }
        for r in audit.read_json(folder / "extraction.json")["records"]:
            rec = r["side"]
            qa = tracking[r["id"], "side"]
            if rec["fingerprint"] != qa["fingerprint"]:
                raise ValueError("Existing side tracking approval differs")
            with np.load(rec["cache"]) as data:
                seq = mask_reviewed_frames(
                    data["image_xy"],
                    data["frame_numbers"],
                    qa.get("unreliable_frame_intervals", []),
                )
            cap = cv2.VideoCapture(annotations[r["id"]]["video_path"])
            seq = three_items.mask_image_border(seq, cap.get(3), cap.get(4))
            cap.release()
            sequences[r["id"]] = seq
    # 앞 배치에서 추가한 두 해군 측면 구간도 같은 특징 정의로 변환한다.
    folder = ROOT / "allfive_round1"
    annotations = {
        r["id"]: r for r in audit.read_json(folder / "score_annotations.json")
    }
    for r in audit.read_json(folder / "extraction.json")["records"]:
        if not r["id"].startswith("held_navy"):
            continue
        with np.load(r["extraction"]["cache"]) as data:
            seq = data["image_xy"].copy()
        cap = cv2.VideoCapture(annotations[r["id"]]["video_path"])
        sequences[r["id"]] = three_items.mask_image_border(
            seq, cap.get(3), cap.get(4)
        )
        cap.release()
    changes = []
    for r in result:
        if r["exercise"] not in ("plank", "squats"):
            continue
        seq = sequences.get(r["id"])
        if seq is None:
            continue
        change = update_note(r, seq, allow_partial_top)
        if change:
            changes.append(change)
    return result, dict(
        records=changes,
        labels_changed=False,
        automatic_features_not_direct_reviews=True,
    )


def fit(rows: list[dict]) -> dict:
    """학습 역할에서만 AI를 학습하고 새 발 비율 단위의 규칙을 보정한다.

    Args:
        rows: train 역할의 동일 관측·직접 등급 입력.

    Returns:
        dict: 항목별 AI와 독립 명시 규칙. 새 발 상승 비율의 기본 경계는
        정상 0.1·부분 0.25이며 오류 메타 그룹이 둘 이상일 때만 보정한다.

    Raises:
        ValueError: 검증 역할 혼입·특징 또는 규칙 경계 계약 위반.
    """
    model = base_fit(rows)
    for key, head in model.items():
        if key[2] not in ("side_foot_change_v1", "side_foot_partial_top_v2"):
            continue
        bucket = [
            (r, r["notes"][2], r["levels"][2])
            for r in rows
            if recovery_train.head_key(r, 2) == key
            and r["notes"][2]["eligible"]
            and r["levels"][2] is not None
        ]
        pairs = collections.Counter((g, r["person"]) for r, n, g in bucket)
        people = collections.Counter(g for g, p in pairs)
        weights = [
            1 / (people[g] * pairs[g, r["person"]]) for r, n, g in bucket
        ]
        prior = (0.25, 0.5, 0.1, 0.25)
        choices = [prior]
        if len(people) >= 2 and min(people.values()) >= 2:
            choices += [
                (g, p, h, j)
                for g in (0.15, 0.25, 0.35)
                for p in (0.4, 0.5, 0.7)
                if g < p
                for h in (0.05, 0.1, 0.15, 0.2)
                for j in (0.2, 0.3, 0.4, 0.5)
                if h < j
            ]
        losses = [
            float(
                np.average(
                    [
                        abs(three_training.rule_grade(n, b) - g)
                        for r, n, g in bucket
                    ],
                    weights=weights,
                )
            )
            for b in choices
        ]
        best = min(
            range(len(choices)),
            key=lambda i: (losses[i], choices[i] != prior, i),
        )
        head.update(
            boundary=choices[best],
            new_heel_measurement_units="fixed_foot_length_ratio",
            training_only_rule_calibration=True,
            rule_calibration=dict(
                default=prior,
                selected=choices[best],
                training_weighted_MAE=losses[best],
            ),
        )
    return model
