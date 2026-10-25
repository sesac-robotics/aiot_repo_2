"""몸통 각도에 처짐·들림 방향을 더한 후보를 학습 분할 내부에서 비교한다.

공통 모델·규칙을 교체하지 않는다. 규칙 점수와 제공자 문자열을 특징이나
정답으로 사용하지 않으며 고정 개발 검증은 후보 선택 이후에만 평가한다.
"""

import collections
import json
import pathlib

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

import audit
import three_input
import three_items
from recovery_balance import weights_for
from recovery_prepare import ROOT
from recovery_train import component_groups
from recovery_trunk_geometry import signed_offsets
from resources import check_resources


def run() -> dict:
    """같은 관측 표본에서 각도·방향 특징과 회귀기를 비교해 별도 저장한다.

    Returns:
        dict: 학습 내부 3폴드 선택·고정 개발 항목/총점 오차·제외 이유.
        선택에 쓴 교차검증 오차는 독립 성능 추정치가 아니다. 실제 새 사람
        최종 시험이나 후보의 공통 모델 승격은 수행하지 않는다.

    Raises:
        OSError: 원본 캐시·학습 기록·모델 출력 접근 실패.
        ValueError: 원본·특징·분할·유효 표본 계약 불일치.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    cv2.setNumThreads(1)
    target = ROOT / "trunk_direction_trial"
    target.mkdir(exist_ok=True)
    comparison = audit.read_json(ROOT / "comparison.json")
    inputs = audit.read_json(ROOT / "fit_input_manifest.json")
    canonical = {
        r["id"]: r
        for r in audit.read_json(
            audit.BASE
            / "output/three-exercise-comparison-20261022/qevd-json/score_annotations.json"
        )
    }
    base_cache = {
        r["id"]: r["paths"]
        for r in audit.read_json(
            audit.BASE
            / "output/three-exercise-comparison-20261022/cache_manifest.json"
        )["records"]
    }
    plan = {
        r["id"]: r for r in audit.read_json(ROOT / "input_plan.json")["records"]
    }
    cache = {
        r["id"]: r["extraction"]
        for r in audit.read_json(ROOT / "extraction.json")["records"]
    }
    ready, excluded = [], []
    for row in inputs:
        if (
            row["exercise"] != "pushups"
            or row["items"][0]["bank"] != "side"
            or not row["items"][0]["eligible"]
        ):
            continue
        check_resources()
        identifier = row["id"]
        if identifier in plan:
            source = plan[identifier]
            receipt = cache[identifier]
            if (
                receipt["config"]["source_sha256"] != source["video_sha256"]
                or receipt["config"]["segment"] != source["segment"]
            ):
                raise ValueError("Trial cache source or segment mismatch")
            path, video = receipt["cache"], source["video_path"]
        else:
            source = canonical[identifier]
            path, video = (
                base_cache[identifier]["side"]["path"],
                source["local_video_path"],
            )
        with np.load(path) as data:
            seq = data["image_xy"].copy()
        cap = cv2.VideoCapture(
            str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        width, height = cap.get(3), cap.get(4)
        cap.release()
        seq = three_items.mask_image_border(seq, width, height)
        if identifier not in plan:
            seq = three_input.apply_framing(seq, source, "side")
        note = three_items.describe("pushups", seq, None)[0]
        if not np.allclose(
            note["feature"], row["items"][0]["feature"], equal_nan=True
        ):
            raise ValueError("Trial angle features differ from fitted inputs")
        offsets = signed_offsets(seq, note["side"])
        valid = offsets[np.isfinite(offsets)]
        if len(valid) < 5 or len(valid) / len(offsets) < 0.8:
            excluded.append(
                dict(id=identifier, reason="direction_geometry_unobserved")
            )
            continue
        angle = row["items"][0]["feature"]
        ready.append(
            dict(
                id=identifier,
                person=row["person_metadata_group"],
                sources=set(row["sources"]),
                recording_groups=row["recording_groups"],
                split=row["split"],
                levels=row["levels"],
                angle=angle,
                direction=angle
                + [
                    float(np.percentile(valid, q))
                    for q in (5, 10, 25, 50, 75, 90, 95)
                ],
                measurement=note["measurement"],
            )
        )
    component_groups(ready)
    train = [
        r for r in ready if r["split"] == "train" and r["levels"][0] is not None
    ]
    val = [
        r
        for r in ready
        if r["split"] == "validation" and r["levels"][0] is not None
    ]
    groups = component_groups(train)
    if len(set(groups)) < 3:
        raise ValueError("Three connected training groups required")

    def fit(rows, policy, floor):
        model = (
            IsotonicRegression(
                increasing=True, out_of_bounds="clip", y_min=0, y_max=2
            )
            if policy == "isotonic"
            else ExtraTreesRegressor(
                n_estimators=128,
                min_samples_leaf=2,
                random_state=221022,
                n_jobs=1,
            )
        )
        model.fit(
            [
                r["measurement"] if policy == "isotonic" else r[policy]
                for r in rows
            ],
            [r["levels"][0] for r in rows],
            sample_weight=weights_for(rows, 0, floor),
        )
        return model

    def infer(model, rows, policy):
        return model.predict(
            [
                r["measurement"] if policy == "isotonic" else r[policy]
                for r in rows
            ]
        )

    losses = {}
    for policy in ("angle", "direction", "isotonic"):
        for floor in (1, 3, 6):
            check_resources()
            errors = collections.defaultdict(list)
            for fitting, held in GroupKFold(3).split(train, groups=groups):
                model = fit([train[int(i)] for i in fitting], policy, floor)
                held_rows = [train[int(i)] for i in held]
                for index, prediction in zip(
                    held, infer(model, held_rows, policy)
                ):
                    errors[groups[int(index)]].append(
                        abs(float(prediction) - train[int(index)]["levels"][0])
                    )
            losses[(policy, floor)] = float(
                np.mean([np.mean(v) for v in errors.values()])
            )
    policy, floor = min(losses, key=lambda k: (losses[k], k[1], k[0]))
    model = fit(train, policy, floor)
    predictions = (
        {r["id"]: float(p) for r, p in zip(val, infer(model, val, policy))}
        if val
        else {}
    )
    original = next(
        x for x in comparison["exercises"] if x["exercise"] == "pushups"
    )
    total_pairs = [
        p
        for p in original["fixed_predictions"]
        if p["id"] in predictions
        and p["target_score"] is not None
        and p["ai_score"] is not None
    ]
    scores = [
        p["ai_score"] + 20 * (predictions[p["id"]] - p["ai_levels"][0])
        for p in total_pairs
    ]
    report = dict(
        code_sha256=audit.digest(pathlib.Path(__file__)),
        fit_input_manifest_sha256=audit.digest(
            ROOT / "fit_input_manifest.json"
        ),
        comparison_sha256=audit.digest(ROOT / "comparison.json"),
        train_records=len(train),
        training_connected_groups=len(set(groups)),
        selection=dict(
            policy=policy,
            class_support_floor=floor,
            inner_group_folds=3,
            validation_targets_used=False,
        ),
        inner_selection_MAE_grade={
            f"{k[0]}_floor{k[1]}": v for k, v in losses.items()
        },
        inner_selection_loss_is_not_unbiased_final_performance=True,
        fixed_development_trunk_records=len(val),
        fixed_development_trunk_MAE_grade=(
            float(
                np.mean(
                    [abs(predictions[r["id"]] - r["levels"][0]) for r in val]
                )
            )
            if val
            else None
        ),
        fixed_development_common_total_records=len(total_pairs),
        fixed_development_baseline_total_MAE_points=(
            float(
                np.mean(
                    [
                        abs(p["ai_score"] - p["target_score"])
                        for p in total_pairs
                    ]
                )
            )
            if total_pairs
            else None
        ),
        fixed_development_trial_total_MAE_points=(
            float(
                np.mean(
                    [
                        abs(score - p["target_score"])
                        for score, p in zip(scores, total_pairs)
                    ]
                )
            )
            if total_pairs
            else None
        ),
        validation_predictions=[
            dict(
                id=r["id"],
                reference=r["levels"][0],
                prediction=predictions[r["id"]],
            )
            for r in val
        ],
        excluded=excluded,
        common_model_promoted=False,
        rule_unchanged=True,
        expert_reference=False,
        independent_final_test=False,
        geometry_code_sha256=audit.digest(
            pathlib.Path(__file__).with_name("recovery_trunk_geometry.py")
        ),
    )
    joblib.dump(
        dict(model=model, selection=report["selection"]),
        target / "model.joblib",
    )
    audit.save_json(target / "comparison.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
