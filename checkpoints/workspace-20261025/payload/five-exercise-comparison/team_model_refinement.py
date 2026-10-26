"""학습 연결 그룹 내부에서만 5종 점수 회귀 후보를 고르고 비교한다."""

import collections
import copy
import json

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

import audit
import recovery_train
from resources import check_resources

ROOT = audit.BASE / "output/five-exercise-recovery-20261022"
OUT = ROOT / "team_refinement"


def candidate(name: str, dimensions: int) -> object:
    """고정 후보 이름으로 직접 등급 회귀기를 만든다.

    Args:
        name: forest2·forest4·forest8·support_monotone 후보 이름.
        dimensions: 입력 특징 차원. 지지 단조 회귀는 22차원을 요구한다.

    Returns:
        object: 아직 학습하지 않은 scikit-learn 회귀기.

    Raises:
        ValueError: 후보 이름·차원 불일치.
    """
    if name == "support_monotone":
        if dimensions != 22:
            raise ValueError("Support clearance requires 22 features")
        return Pipeline([
            ("select", ColumnTransformer([("physical", "passthrough", [0, 7, 14])])),
            ("regressor", HistGradientBoostingRegressor(
                max_iter=80, max_leaf_nodes=5, min_samples_leaf=3,
                monotonic_cst=[1, 1, 1], l2_regularization=1,
                random_state=221023)),
        ])
    if name not in ("forest2", "forest4", "forest8"):
        raise ValueError("Unknown model candidate")
    return ExtraTreesRegressor(n_estimators=128,
        min_samples_leaf=int(name[-1]), random_state=221022,
        max_features=1.0 if name == "forest2" else 0.7, n_jobs=1)


def weights(labels: np.ndarray, groups: list[str]) -> np.ndarray:
    """반복 구간이 많은 연결 그룹이 등급 회귀를 독점하지 않게 가중한다.

    Args:
        labels: 직접 초안 등급 0~2 배열.
        groups: 참여자·원본·동일 수행자 가능성을 연결한 그룹 ID.

    Returns:
        np.ndarray: 등급·그룹별 총합이 균형인 양수 가중치.
    """
    counts = collections.Counter(zip(labels, groups))
    people = collections.Counter(g for g, _ in counts)
    return np.asarray([1 / (people[y] * counts[y, g]) for y, g in zip(labels, groups)])


def refine() -> dict:
    """검증 라벨을 수치 학습에 넣지 않고 후보·선택 진단을 별도 저장한다.

    Returns:
        dict: 운동별 학습 내부 선택 내역. 고정 개발 비교·최종 채택은 별도로
        수행한다. 내부 선택 CV를 독립 최종 시험 성능으로 표시하지 않는다.

    Raises:
        ValueError: 원본·참여자 역할 누출 또는 입력 형상 불일치.
        OSError: 기존 입력·모델·후보 저장 실패.
        RuntimeError: 자원 보호 기준 위반.
    """
    OUT.mkdir(exist_ok=True)
    rows = audit.read_json(ROOT / "held_round2/candidate_input_manifest.json")
    rows = [r for r in rows if r["exercise"] != "squats"] + audit.read_json(ROOT / "squat_partial_top_trial/candidate_input_manifest.json")
    recovery_train.component_groups(rows)
    records = []
    for exercise in ("pushups", "squats", "shoulder", "plank", "ytw"):
        check_resources()
        source = ROOT / ("squat_partial_top_trial/squats_candidate.joblib" if exercise == "squats" else f"held_round2/models/{exercise}.joblib")
        model = copy.deepcopy(joblib.load(source))
        training = [r for r in rows if r["exercise"] == exercise and r["split"] == "train"]
        linked = recovery_train.component_groups(training)
        for key, head in model.items():
            index = int(key) if exercise == "ytw" else next(i for i in range(4) if training[0]["items"][i]["item"] == key[1])
            bucket = [(r, g) for r, g in zip(training, linked)
                if r["items"][index]["eligible"] and r["levels"][index] is not None
                and (exercise == "ytw" or r["items"][index]["bank"] == key[2])]
            if len({g for _, g in bucket}) < 3 or len({r['levels'][index] for r, _ in bucket}) < 2:
                continue
            x = np.asarray([r["items"][index]["feature"] for r, _ in bucket])
            y = np.asarray([r["levels"][index] for r, _ in bucket])
            groups = [g for _, g in bucket]
            names = ["forest2", "forest4", "forest8"]
            if exercise == "plank" and index == 2 and x.shape[1] == 22:
                names.append("support_monotone")
            losses = {}
            for name in names:
                predicted = np.full(len(y), np.nan)
                for fitting, held in GroupKFold(3).split(x, groups=groups):
                    estimator = candidate(name, x.shape[1])
                    weight = weights(y[fitting], [groups[int(i)] for i in fitting])
                    kwargs = {"regressor__sample_weight": weight} if name == "support_monotone" else {"sample_weight": weight}
                    estimator.fit(x[fitting], y[fitting], **kwargs)
                    predicted[held] = estimator.predict(x[held])
                losses[name] = float(np.average(abs(predicted-y), weights=weights(y, groups)))
            selected = min(names, key=lambda name: (losses[name], names.index(name)))
            estimator = candidate(selected, x.shape[1])
            weight = weights(y, groups)
            kwargs = {"regressor__sample_weight": weight} if selected == "support_monotone" else {"sample_weight": weight}
            estimator.fit(x, y, **kwargs)
            head["model"] = estimator
            record = dict(exercise=exercise, head=str(key), selected=selected,
                training_selection_cv_MAE=losses, records=len(y),
                connected_groups=len(set(groups)), validation_used_for_numeric_fitting=False)
            records.append(record)
            audit.save_json(OUT / "selection.json", dict(records=records, completed=False))
        joblib.dump(model, OUT / f"{exercise}.joblib")
    result = dict(records=records, completed=True, independent_final_test=False,
        selection_cv_is_not_final_performance=True)
    audit.save_json(OUT / "selection.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(refine(), ensure_ascii=False))
