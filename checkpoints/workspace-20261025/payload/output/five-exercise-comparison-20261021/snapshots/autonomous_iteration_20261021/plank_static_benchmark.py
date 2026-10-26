"""정지 몸통 정렬 초안으로 AI와 고정 규칙을 참여자별 교차 검증한다.

기본 플랭크의 지속·지지·총점을 대신하지 않는 별도 개발 실험이다.
동일한 관측 관절을 사용하고 제공자 상태·참여자 ID를 특징에 넣지 않는다.
"""

import collections
import time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

import audit
from resources import check_resources


def pose_features(xy: np.ndarray, confidence: np.ndarray) -> dict | None:
    """한 화면의 관측된 어깨·골반·발목으로 정규화 특징을 만든다.

    Args:
        xy: 33×2 원본 픽셀 좌표. 미관측은 NaN.
        confidence: 33×2 visibility·presence, 범위 0~1.

    Returns:
        dict | None: 몸통 길이로 나눈 발목의 축 방향·수직 절대 성분,
        골반 내각(도), 선택 측면. 두 측면 모두 누락·퇴화이면 None.

    Raises:
        ValueError: 배열 형상이 계약과 다를 때.
    """
    if xy.shape != (33, 2) or confidence.shape != (33, 2):
        raise ValueError("Expected 33 x 2 coordinates and confidence")
    candidates = []
    for side, indices in (("left", [11, 23, 27]), ("right", [12, 24, 28])):
        points = xy[indices]
        quality = confidence[indices]
        if not np.isfinite(points).all() or not np.isfinite(quality).all():
            continue
        if np.min(quality) < 0.5:
            continue
        shoulder, hip, ankle = points
        torso, leg = shoulder - hip, ankle - hip
        length, leg_length = np.linalg.norm(torso), np.linalg.norm(leg)
        if min(length, leg_length) <= 1:
            continue
        axis = torso / length
        longitudinal = float(np.dot(leg, axis) / length)
        perpendicular = float(abs(leg[0] * axis[1] - leg[1] * axis[0]) / length)
        angle = float(
            np.degrees(
                np.arccos(
                    np.clip(np.dot(torso, leg) / (length * leg_length), -1, 1)
                )
            )
        )
        candidates.append(
            dict(
                side=side,
                quality=float(np.min(quality)),
                features=[longitudinal, perpendicular],
                angle_deg=angle,
            )
        )
    return (
        max(candidates, key=lambda row: row["quality"]) if candidates else None
    )


def check_groups(records: list[dict]) -> None:
    """같은 이미지 해시가 서로 다른 참여자에 배정됐는지 검사한다.

    Args:
        records: id·person_token·image_sha256을 가진 이미지 목록.

    Returns:
        None: ID와 이미지의 참여자 그룹이 유효하면 반환한다.

    Raises:
        ValueError: ID 중복 또는 동일 이미지의 그룹 충돌.
        KeyError: 필수 식별 필드가 없는 경우.
    """
    ids, hashes = set(), {}
    for row in records:
        if row["id"] in ids:
            raise ValueError("Duplicate record ID")
        ids.add(row["id"])
        previous = hashes.setdefault(row["image_sha256"], row["person_token"])
        if previous != row["person_token"]:
            raise ValueError("Duplicate source crosses person groups")


def run() -> dict:
    """라벨 해시를 검증하고 관측된 참여자별 개발 폴드와 결과를 저장한다.

    Returns:
        dict: 동일 관측 집합의 등급·40점 항목 오차, 처리 초, 폴드,
        실패와 예측. 100점 총점·최종 시험·전문가 정확도는 생성하지 않는다.

    Raises:
        ValueError: 동결 라벨·캐시 불일치 또는 학습 가능한 그룹 부족.
        OSError: 입력·결과·실험 모델 접근 실패.
        RuntimeError: 자원 보호 기준 초과.
    """
    folder = audit.OUT / "plank_static_review"
    protocol = audit.read_json(folder / "frozen_protocol.json")
    if audit.digest(folder / "direct_labels.json") != protocol["label_sha256"]:
        raise ValueError("Frozen labels changed")
    if (
        audit.digest(audit.BASE / "five-exercise-comparison/rule_score.py")
        != protocol["rule_code_sha256"]
    ):
        raise ValueError("Common reference rule changed")
    labels = audit.read_json(folder / "direct_labels.json")["records"]
    check_groups(labels)
    cache = np.load(folder / "heavy_static.npz", allow_pickle=False)
    metadata = audit.read_json(folder / "heavy_static.json")
    if audit.digest(folder / "heavy_static.npz") != metadata["cache_sha256"]:
        raise ValueError("Cache hash mismatch")
    by_id = {str(value): i for i, value in enumerate(cache["ids"])}
    if len(by_id) != len(cache["ids"]) or set(by_id) != {
        r["id"] for r in labels
    }:
        raise ValueError("Cache IDs are duplicated or do not match labels")
    if str(cache["schema"]) != "heavy-static-IMAGE-v1":
        raise ValueError("Unexpected cache schema")
    rows, features, target, people, eligible = [], [], [], [], []
    rule_times, ai_times, folds = [], [], []
    for label in labels:
        index = by_id[label["id"]]
        if str(cache["source_sha256"][index]) != label["image_sha256"]:
            raise ValueError("Image label/cache mismatch")
        start = time.perf_counter()
        pose = pose_features(
            cache["image_xy"][index], cache["confidence"][index]
        )
        feature_seconds = time.perf_counter() - start
        start = time.perf_counter()
        rule = (
            None
            if pose is None
            else (
                2
                if pose["angle_deg"] >= protocol["rule_good_min_angle_deg"]
                else (
                    1
                    if pose["angle_deg"]
                    >= protocol["rule_partial_min_angle_deg"]
                    else 0
                )
            )
        )
        rule_times.append(time.perf_counter() - start)
        level = label["draft_levels"][0]
        row = dict(
            id=label["id"],
            person_token=label["person_token"],
            image_path=label["image_path"],
            image_sha256=label["image_sha256"],
            draft_grade=level,
            rule_grade=rule,
            ai_oof_grade=None,
            train_mean_grade=None,
            pose=pose,
            feature_seconds=feature_seconds,
            total_score=None,
            full_score_hold_reasons=label["other_hold_reasons_ko"],
        )
        if level is not None and pose is not None:
            eligible.append(len(rows))
            features.append(pose["features"])
            target.append(level)
            people.append(label["person_token"])
        rows.append(row)
    x, y, groups = np.asarray(features), np.asarray(target), np.asarray(people)
    if len(set(people)) < 4 or len(y) < 16 or len(set(target)) < 2:
        raise ValueError("Insufficient observable grouped development labels")
    for person in sorted(set(people)):
        check_resources()
        test, train = groups == person, groups != person
        if len(set(groups[train])) < 3:
            raise ValueError("Insufficient training groups")
        model = ExtraTreesRegressor(**protocol["AI_parameters"])
        start = time.perf_counter()
        model.fit(x[train], y[train])
        fit_seconds = time.perf_counter() - start
        predictions = []
        for index in np.flatnonzero(test):
            start = time.perf_counter()
            prediction = float(model.predict(x[index : index + 1])[0])
            ai_times.append(time.perf_counter() - start)
            rows[eligible[index]]["ai_oof_grade"] = prediction
            rows[eligible[index]]["train_mean_grade"] = float(np.mean(y[train]))
            predictions.append(prediction)
        folds.append(
            dict(
                held_person=person,
                train_people=sorted(set(groups[train].tolist())),
                train_images=int(train.sum()),
                test_images=int(test.sum()),
                fit_seconds=fit_seconds,
                grade_mae=float(
                    np.mean(np.abs(np.asarray(predictions) - y[test]))
                ),
            )
        )
    common = [r for r in rows if r["ai_oof_grade"] is not None]
    metrics = {}
    for method in ("ai_oof_grade", "rule_grade", "train_mean_grade"):
        errors = np.asarray([r[method] - r["draft_grade"] for r in common])
        metrics[method] = dict(
            grade_mae=float(np.mean(abs(errors))),
            contribution_40_mae=float(np.mean(abs(errors)) * 20),
            grade_bias=float(np.mean(errors)),
        )
    final_model = ExtraTreesRegressor(**protocol["AI_parameters"])
    final_model.fit(x, y)
    model_path = folder / "experimental_trunk_item_model.joblib"
    joblib.dump(
        dict(
            model=final_model,
            schema="plank-static-trunk-two-coordinates-v1",
            feature_names=[
                "ankle_longitudinal_torso_ratio",
                "ankle_perpendicular_abs_torso_ratio",
            ],
            total_score_supported=False,
            label_sha256=protocol["label_sha256"],
        ),
        model_path,
    )
    result = dict(
        protocol=protocol,
        source_images=len(rows),
        directly_drafted_trunk_labels=sum(
            r["draft_grade"] is not None for r in rows
        ),
        observable_images=sum(r["pose"] is not None for r in rows),
        common_evaluated_images=len(common),
        common_evaluation_fraction=len(common) / len(rows),
        common_fraction_among_labeled=len(common)
        / sum(r["draft_grade"] is not None for r in rows),
        grade_counts=dict(
            collections.Counter(str(r["draft_grade"]) for r in common)
        ),
        metrics=metrics,
        folds=folds,
        rows=rows,
        model_path=str(model_path),
        model_sha256=audit.digest(model_path),
        independent_final_test_count=0,
        expert_labels=0,
        full_plank_training_records_added=0,
        temporal_consistency_tested=False,
        extraction_mode="IMAGE",
        same_source_same_common_subset=True,
        timing_seconds=dict(
            heavy_initialization=metadata["initialization_seconds"],
            heavy_decode_resize_inference_median=float(
                np.median(metadata["decode_resize_inference_seconds"])
            ),
            feature_median=float(
                np.median([r["feature_seconds"] for r in rows])
            ),
            rule_compute_median=float(np.median(rule_times)),
            ai_single_prediction_median=float(np.median(ai_times)),
        ),
        limitations_ko=[
            "비전문가 실험 초안에 대한 참여자 토큰 분리 개발 검증이며 독립 최종 시험이 아님",
            "사선 2D 정렬만 비교하며 실제 하중·기구 지지·시간 지속은 알 수 없음",
            "참여자 토큰과 실제 인물의 일치 여부는 외부 확인 전",
            "한 촬영 환경의 단일 구도만 사용하여 구도·장치 일반화 미검증",
            "관측 실패를 점수로 채우지 않으며 40점 항목 기여분을 100점 총점으로 환산하지 않음",
        ],
    )
    audit.save_json(folder / "comparison.json", result)
    return result


if __name__ == "__main__":
    report = run()
    print(
        {
            key: report[key]
            for key in [
                "observable_images",
                "common_evaluated_images",
                "grade_counts",
                "metrics",
                "timing_seconds",
            ]
        }
    )
