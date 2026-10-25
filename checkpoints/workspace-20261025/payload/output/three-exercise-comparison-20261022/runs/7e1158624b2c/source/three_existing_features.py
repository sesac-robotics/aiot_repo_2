"""보유 정면 자료에서 구도가 맞는 항목만 학습 후보 특징으로 연결한다.

측면 몸통·하강 범위·접지 특징을 정면에 적용하지 않는다. 푸시업의
관측 가능한 팔 펴기와 스쿼트의 정면 좌우 균형만 분리 저장한다.
"""

import collections
import json

import cv2
import numpy as np

import audit
import three_items
from three_existing_review import ROOT


def run() -> dict:
    """Heavy 좌표와 직접 라벨을 동일 구간·구도별 항목 특징으로 연결한다.

    Returns:
        dict: 라벨·특징이 함께 유효한 항목 수, 관측 실패와 미구현 항목.
        특징의 각도 단위는 도이며 등급·점수·제공자 문자열은 입력에 없다.
        오류 등급의 사람 다양성이 없어 새 모델 학습은 수행하지 않는다.

    Raises:
        OSError: 좌표·원본·라벨·출력 접근 실패.
        ValueError: 캐시 구간·원본 해시 불일치 또는 좌표 형상 오류.
    """
    labels = {
        x["id"]: x for x in audit.read_json(ROOT / "score_annotations.json")
    }
    prepared = audit.read_json(ROOT / "heavy_preparation.json")
    if not prepared["completed"]:
        raise ValueError("Incomplete extraction")
    records = []
    distribution = collections.defaultdict(collections.Counter)
    for item in prepared["records"]:
        row = labels[item["id"]]
        extraction = item["extraction"]
        config = extraction["config"]
        if (
            config["source_sha256"] != row["video_sha256"]
            or config["segment"] != row["segment"]
        ):
            raise ValueError("Cache provenance mismatch")
        with np.load(extraction["cache"]) as cache:
            sequence = cache["image_xy"].copy()
        cap = cv2.VideoCapture(row["video_path"])
        width, height = cap.get(3), cap.get(4)
        cap.release()
        sequence = three_items.mask_image_border(sequence, width, height)
        exercise = row["exercise"]
        if exercise == "pushups":
            descriptions = three_items.describe(exercise, sequence, None)
            allowed = {"return_extension"}
        elif exercise == "squats":
            descriptions = three_items.describe(exercise, None, sequence)
            allowed = {"trunk_balance"}
        else:
            descriptions = three_items.apply_view_policy(
                exercise,
                three_items.describe(exercise, sequence, None),
                row["required_input_views"],
            )
            allowed = {"overhead_reach", "lower_return", "bilateral_timing"}
        notes = []
        for index, (name, grade) in enumerate(
            zip(
                row["assessment"]["criterion_ids"],
                row["assessment"]["criterion_levels"],
            )
        ):
            description = descriptions[index]
            ready = (
                grade is not None
                and name in allowed
                and description["eligible"]
            )
            note = dict(
                criterion=name,
                direct_grade=grade,
                eligible_for_item_feature_dataset=ready,
                reason=(
                    None
                    if ready
                    else (
                        "label_unavailable"
                        if grade is None
                        else (
                            "front_view_feature_not_implemented"
                            if name not in allowed
                            else description["holds"]
                        )
                    )
                ),
                feature=description["feature"] if ready else None,
                feature_units=(
                    "angle_degrees_quantiles_plus_observed_fraction"
                    if ready
                    else None
                ),
            )
            notes.append(note)
            if ready:
                distribution[exercise + "/" + name][str(grade)] += 1
        records.append(
            dict(
                id=row["id"],
                exercise=exercise,
                person=row["person"],
                split=row["split"],
                view=row["required_input_views"][0],
                video_sha256=row["video_sha256"],
                segment=row["segment"],
                items=notes,
            )
        )
    report = dict(
        records=records,
        distribution=dict(distribution),
        features_do_not_contain_reference_or_provider_text=True,
        front_features_not_mixed_into_side_models=True,
        active_models_updated=False,
        why_not_fit="추가 특징의 등급이 모두 2인 항목은 오류 분류를 학습할 수 없다. 기존 자료에서 오류 사례 선정·라벨 보강을 계속해야 한다.",
    )
    audit.save_json(ROOT / "front_item_features.json", report)
    return dict(
        records=len(records),
        eligible_items=sum(
            x["eligible_for_item_feature_dataset"]
            for r in records
            for x in r["items"]
        ),
        distribution=dict(distribution),
    )


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
