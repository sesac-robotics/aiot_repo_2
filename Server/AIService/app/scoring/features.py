"""관측 구도별 Strategy로 학습 때와 같은 점수 입력 특징을 만든다."""

import numpy as np

from app.scoring import geometry, held_geometry, plank_item_observation
from app.scoring import three_items, ytw_features
from app.scoring.sequence import PoseSequence

DEFINITIONS = {
    "pushups": (
        ["trunk_line", "descent_range", "return_extension", "support"],
        [40, 30, 20, 10],
    ),
    "squats": (
        ["descent_range", "return_extension", "knee_foot", "trunk_balance"],
        [30, 20, 30, 20],
    ),
    "shoulder": (
        [
            "overhead_reach",
            "lower_return",
            "bilateral_timing",
            "frontal_trunk_hold",
        ],
        [30, 25, 25, 20],
    ),
    "plank": (plank_item_observation.ITEMS, plank_item_observation.WEIGHTS),
    "ytw": (
        ["y_reach", "t_reach", "w_shape", "hinge_and_trunk_hold"],
        [30, 25, 25, 20],
    ),
}


def describe(
    exercise: str,
    primary: PoseSequence,
    model: dict,
    *,
    view: str,
    front: PoseSequence | None = None,
    phase_windows: dict | None = None,
    projection_reviewed: bool = False,
) -> list[dict]:
    """가시성·구도·자동 단계의 교집합으로 네 항목을 반환한다.

    Args:
        exercise: DEFINITIONS의 운동 코드.
        primary: 지정 구도의 10Hz 원본 픽셀 관절.
        model: 특징 버전이 함께 고정된 학습 모델.
        view: 직접 확인한 side·front·rear·front_oblique 구도.
        front: 스쿼트의 같은 수행·정렬 확인을 마친 정면 관절.
        phase_windows: YTW 단계별 직접 확인 원본 프레임 구간. 점수가 아니다.
        projection_reviewed: Y·전경 화면 대리값의 촬영 구도 직접 확인 여부.

    Returns:
        list[dict]: 네 항목의 관측·특징·기하 측정·미판정 사유. 미관측은
        보간하지 않고 구도가 다른 특징을 같은 모델 묶음으로 합치지 않는다.

    Raises:
        ValueError: 운동·촬영 구도·단계 구간 형상 불일치.
    """
    if exercise not in DEFINITIONS or view not in (
        "side",
        "front",
        "rear",
        "front_oblique",
    ):
        raise ValueError("Unsupported exercise or reviewed view")
    seq = three_items.mask_image_border(
        primary.image_xy, primary.width, primary.height
    )
    frontal = (
        three_items.mask_image_border(front.image_xy, front.width, front.height)
        if front is not None
        else None
    )
    if exercise in ("pushups", "squats", "shoulder"):
        if exercise == "shoulder" and view == "side":
            raise ValueError("Shoulder model requires a frontal-plane view")
        side = seq if view == "side" else None
        notes = three_items.describe(
            exercise,
            seq,
            (
                frontal
                if front is not None
                else (seq if exercise == "squats" and view != "side" else None)
            ),
        )
        # 기존 정면 푸시업 특징도 팔 관절만으로 같은 수식으로 측정한다.
        if exercise == "pushups" and side is None:
            notes = three_items.describe(exercise, seq, None)
        for i, note in enumerate(notes):
            note["bank"] = (
                ("front_oblique" if view == "front_oblique" else "front")
                if exercise == "shoulder"
                else (
                    "front"
                    if i == 3 and exercise == "squats" or side is None
                    else "side"
                )
            )
            if (
                side is None
                and exercise in ("pushups", "squats")
                and i not in ({1, 2} if exercise == "pushups" else {0, 1, 3})
            ):
                note["eligible"] = False
                note["holds"].append("item_requires_side_view")
        notes = three_items.apply_view_policy(exercise, notes, [view])
        if exercise == "squats" and side is not None and notes[2]["eligible"]:
            banks = {
                k[2]
                for k in model
                if isinstance(k, tuple) and k[1] == "knee_foot"
            }
            bank = next(
                (
                    b
                    for b in ("side_foot_partial_top_v2", "side_foot_change_v1")
                    if b in banks
                ),
                None,
            )
            if bank:
                result = held_geometry.squat_foot_change(
                    seq, bank.endswith("v2")
                )
                notes[2].update(
                    bank=bank,
                    eligible=result["eligible"],
                    feature=(
                        notes[2]["feature"][:-1] + result["feature"]
                        if result["eligible"]
                        else None
                    ),
                )
                notes[2]["holds"].extend(result["holds"])
    elif exercise == "plank":
        if view != "side":
            raise ValueError("Released plank model requires reviewed side view")
        assessment = plank_item_observation.assess(
            seq, primary.duration_seconds, [True] * 4
        )
        channels = geometry.channels(seq)
        notes = []
        for i, item in enumerate(assessment["items"]):
            values = channels[
                item["side"] + "_" + plank_item_observation.CHANNELS[i]
            ]
            valid = values[np.isfinite(values)]
            notes.append(
                dict(
                    item=item["criterion_id"],
                    eligible=item["eligible"] and len(valid) > 0,
                    feature=(
                        np.percentile(
                            valid, [5, 10, 25, 50, 75, 90, 95]
                        ).tolist()
                        if len(valid)
                        else None
                    ),
                    measurement=item["measured"],
                    higher_is_better=i in (0, 2),
                    bank="side_video",
                    rule_grade=item["rule_grade"],
                    holds=item["reasons"],
                )
            )
        if (
            any(
                isinstance(k, tuple) and k[2] == "side_support_clearance_v1"
                for k in model
            )
            and notes[2]["eligible"]
        ):
            result = held_geometry.plank_clearance(seq)
            notes[2].update(
                bank="side_support_clearance_v1",
                eligible=result["eligible"],
                feature=result["feature"],
                measurement=result["measurement"],
                rule_grade=held_geometry.plank_support_grade(
                    result["measurement"]
                ),
            )
            notes[2]["holds"].extend(result["holds"])
    else:
        if view not in ("front", "rear"):
            raise ValueError(
                "YTW projection experiment requires frontal-plane view"
            )
        notes = ytw_features.describe_ytw(seq)
        refined = ytw_features.refined_w(seq)
        if refined:
            notes[2].update(refined, eligible=True)
        windows = phase_windows or {}
        for i, note in enumerate(notes):
            note.update(bank="front_projected", holds=[])
            if i < 3 and note["item"] in windows:
                interval = windows[note["item"]]
                if i == 2 and not ytw_features.phase_matches(
                    note.get("window"), primary.frame_numbers, interval
                ):
                    absence = ytw_features.observed_absence(seq)
                    if absence and ytw_features.phase_matches(
                        absence["window"], primary.frame_numbers, interval
                    ):
                        note.update(
                            eligible=True,
                            feature=absence["feature"],
                            window=absence["window"],
                        )
                if not ytw_features.phase_matches(
                    note.get("window"), primary.frame_numbers, interval
                ):
                    note["eligible"] = False
                    note["holds"].append("phase_interval_mismatch")
            if i in (0, 3) and not projection_reviewed:
                note["eligible"] = False
                note["holds"].append("projection_view_review_required")
    for note in notes:
        if (
            note.get("feature") is None
            or not np.isfinite(note["feature"]).all()
        ):
            note["eligible"] = False
            note.setdefault("holds", []).append(
                "finite_required_geometry_unavailable"
            )
        note["eligible"] = bool(note["eligible"])
    return notes
