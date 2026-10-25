"""정상·오류 및 참여자 비중을 학습 폴드 안에서만 맞추는 AI 실험이다."""

import collections

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import audit
import benchmark
import plank_rule_trial
import rule_score
from resources import check_resources


def fitting_weights(labels: list[dict], people: list[str]) -> np.ndarray:
    """학습 폴드의 정상·오류 클래스와 클래스 내 참여자 비중을 맞춘다.

    Args:
        labels: 해당 학습 폴드의 score가 있는 직접 초안 목록. 점수 단위 0~100.
        people: 같은 순서의 참여자 그룹 ID. 보류 참여자를 포함하면 안 된다.

    Returns:
        np.ndarray: 평균 1인 N개 가중치. 정상은 100점 초안, 오류는 그 미만
        초안이며 입력 특징·새 정답을 만들지 않는다. 한 클래스뿐이어도 유효하다.

    Raises:
        ValueError: 빈 목록 또는 길이 불일치.
    """
    if not labels or len(labels) != len(people):
        raise ValueError("Nonempty matching fit labels and people required")
    classes = [int(row["score"] < 100) for row in labels]
    counts = collections.Counter(zip(classes, people))
    groups = {c: len({p for k, p in counts if k == c}) for c in set(classes)}
    weights = np.asarray(
        [
            1 / (len(groups) * groups[c] * counts[(c, p)])
            for c, p in zip(classes, people, strict=True)
        ]
    )
    return weights / weights.mean()


def run() -> dict:
    """기존 10개 개발 자료에서 두 학습 방식과 새 규칙안을 교차 비교한다.

    Returns:
        dict: 참여자 제외 예측·라벨별 오차·새 규칙 결과와 실험 모델 경로.
        제외 참여자의 라벨로 가중치·결측 대체값을 계산하지 않는다. 원래
        공통 모델은 교체하지 않고 새 외부 자료를 이 학습에 넣지 않는다.

    Raises:
        OSError: 입력·관절·모델·결과 접근 실패.
        ValueError: 관측 보류 또는 참여자·원본 누출.
        RuntimeError: 자원 보호 중단.
    """
    check_resources()
    targets = {r["id"]: r for r in audit.read_json(audit.OUT / "labels.json")}
    eligible_ids = {
        r["id"]
        for r in audit.read_json(audit.OUT / "comparison.json")["results"][
            "plank"
        ]["rule_only_previews"]
    }
    data = []
    for row in audit.read_json(audit.OUT / "development_manifest.json"):
        if (
            row["id"] not in eligible_ids
            or row["exercise"] != "plank"
            or row["original_split"] != "train"
            or not row["eligible_label"]
        ):
            continue
        primary, front, holds = benchmark.load_input(row)
        if holds:
            continue
        x, names = benchmark.input_features(primary, front)
        data.append(
            dict(
                manifest=row,
                label=targets[row["id"]],
                x=x,
                names=names,
                rule=rule_score.evaluate(
                    "plank",
                    primary,
                    None,
                    targets[row["id"]]["criterion_ids"],
                    [40, 25, 20, 15],
                ),
                rule_trial=plank_rule_trial.evaluate(primary),
            )
        )
    people = sorted({r["manifest"]["person"] for r in data})
    reports = []
    folder = audit.OUT / "plank_balance_trial"
    folder.mkdir(exist_ok=True)
    for variant in ["unweighted_leaf2", "balanced_leaf2", "balanced_leaf1"]:
        predictions = []
        for held in people:
            fitting = [r for r in data if r["manifest"]["person"] != held]
            testing = [r for r in data if r["manifest"]["person"] == held]
            hashes = lambda rows: {
                r["manifest"].get(k)
                for r in rows
                for k in ["video_sha256", "source_original_sha256"]
                if r["manifest"].get(k)
            }
            if hashes(fitting) & hashes(testing):
                raise ValueError("Person/source leakage")
            model = make_pipeline(
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
                ExtraTreesRegressor(
                    n_estimators=128,
                    min_samples_leaf=1 if variant.endswith("leaf1") else 2,
                    random_state=21021,
                    n_jobs=1,
                ),
            )
            fit_kwargs = {}
            if variant.startswith("balanced"):
                fit_kwargs["extratreesregressor__sample_weight"] = (
                    fitting_weights(
                        [r["label"] for r in fitting],
                        [r["manifest"]["person"] for r in fitting],
                    )
                )
            model.fit(
                np.asarray([r["x"] for r in fitting]),
                np.asarray([r["label"]["criterion_levels"] for r in fitting]),
                **fit_kwargs,
            )
            for row, levels in zip(
                testing, model.predict([r["x"] for r in testing]), strict=True
            ):
                score = float(
                    np.dot(np.clip(levels, 0, 2), [40, 25, 20, 15]) / 2
                )
                predictions.append(
                    dict(
                        id=row["manifest"]["id"],
                        held_person=held,
                        direct_score=row["label"]["score"],
                        ai_score=score,
                        ai_levels=levels.tolist(),
                        original_rule_score=row["rule"]["score"],
                        trial_rule_score=row["rule_trial"]["score"],
                        trial_rule_levels=row["rule_trial"]["levels"],
                        trial_evidence=row["rule_trial"]["evidence"]["trial"],
                    )
                )
        metrics = {}
        for kind in ["all", "normal", "error"]:
            selected = [
                r
                for r in predictions
                if kind == "all"
                or (r["direct_score"] == 100) == (kind == "normal")
            ]
            metrics[kind] = {}
            for method in [
                "ai_score",
                "original_rule_score",
                "trial_rule_score",
            ]:
                paired = [r for r in selected if r[method] is not None]
                metrics[kind][method] = dict(
                    samples=len(paired),
                    MAE=(
                        float(
                            np.mean(
                                [
                                    abs(r[method] - r["direct_score"])
                                    for r in paired
                                ]
                            )
                        )
                        if paired
                        else None
                    ),
                )
        reports.append(
            dict(variant=variant, records=predictions, metrics=metrics)
        )
        model = make_pipeline(
            SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            ),
            ExtraTreesRegressor(
                n_estimators=128,
                min_samples_leaf=1 if variant.endswith("leaf1") else 2,
                random_state=21021,
                n_jobs=1,
            ),
        )
        kwargs = (
            {}
            if not variant.startswith("balanced")
            else dict(
                extratreesregressor__sample_weight=fitting_weights(
                    [r["label"] for r in data],
                    [r["manifest"]["person"] for r in data],
                )
            )
        )
        model.fit(
            np.asarray([r["x"] for r in data]),
            np.asarray([r["label"]["criterion_levels"] for r in data]),
            **kwargs,
        )
        joblib.dump(
            dict(
                model=model,
                names=data[0]["names"],
                train_ids=[r["manifest"]["id"] for r in data],
                expert_validated=False,
            ),
            folder / f"{variant}.joblib",
        )
    report = dict(
        variants=reports,
        train_records=len(data),
        train_people=len(people),
        adopted=False,
        common_model_modified=False,
        common_rule_modified=False,
        independent_final_test=False,
        expert_validated=False,
        rule_trial_sha256=audit.digest(
            audit.BASE / "five-exercise-comparison/plank_rule_trial.py"
        ),
        source_label_sha256=audit.digest(audit.OUT / "labels.json"),
    )
    audit.save_json(audit.OUT / "plank_balance_trial.json", report)
    return report


if __name__ == "__main__":
    print([(r["variant"], r["metrics"]) for r in run()["variants"]])
