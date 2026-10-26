"""QEVD 4종의 클립 오류 분류 실험을 Heavy 전용로 실행한다.

원본·기존 실험 코드는 변경하지 않는다. 특징 캐시와 모델·시험 결과는
이 폴더에 저장한다. 모델별 독립 학습이며 0~100점 회귀를 수행하지 않는다.
"""

import collections
import hashlib
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline

import catalog
import pose_features

LOGGER = logging.getLogger(__name__)


def save_json(path: Path, value: object) -> None:
    """UTF-8 JSON을 임시 파일로 쓴 후 교체한다.

    Args:
        path: 저장할 경로.
        value: JSON으로 직렬화 가능한 값. NaN은 허용하지 않는다.
    Raises:
        OSError: 디스크 쓰기 실패.
        ValueError: NaN 등 직렬화할 수 없는 값이 있는 경우.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temp.replace(path)


def fit_exercise(model: str, exercise: str, items: list, folder: Path) -> dict:
    """오류마다 명시 기준군과 비교해 학습하고 독립 사람 시험을 평가한다.

    Args:
        model: 관절 추출 모델 이름.
        exercise: 운동 키.
        items: 전체 영상 주석. 오류 및 기준군 정보 포함.
        folder: 모델별 결과 경로.
    Returns:
        오류별 검증·시험 지표와 시험 영상 예측. 표본 부족은 unsupported.
    Raises:
        OSError: 캐시 또는 모델 저장 실패.
    """
    rows, features = [], []
    for item in items:
        if item["exercise"] != exercise:
            continue
        cached = np.load(folder / "features" / f"{item['id']}.npz")
        if cached["X"].size:
            rows.append(item)
            features.append(cached["X"])
    profile = catalog.PROFILES[exercise]
    result = dict(name=profile["name"], extraction=model, errors={}, demos=[])
    if not features:
        result["status"] = "no_valid_pose"
        return result
    x = np.asarray(features)
    splits = np.array([r["split"] for r in rows])
    predictions = {r["id"]: [] for r in rows if r["split"] == "test"}
    for error, (name, advice) in profile["errors"].items():
        eligible = np.array(
            [r["reference"] or error in r["errors"] for r in rows]
        )
        y = np.array([int(error in r["errors"]) for r in rows])
        masks = {
            s: eligible & (splits == s) for s in ("train", "validation", "test")
        }
        counts = {
            s: {str(c): int(sum(m & (y == c))) for c in (0, 1)}
            for s, m in masks.items()
        }
        record = dict(name=name, advice=advice, counts=counts)
        if any(min(v.values()) < 5 for v in counts.values()):
            record["status"] = "unsupported_insufficient_samples"
            result["errors"][error] = record
            continue
        classifier = make_pipeline(
            SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            ),
            ExtraTreesClassifier(
                n_estimators=160,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=37,
                n_jobs=4,
            ),
        )
        classifier.fit(x[masks["train"]], y[masks["train"]])
        validation_prob = classifier.predict_proba(x[masks["validation"]])[:, 1]
        thresholds = [0.35, 0.5, 0.65]
        threshold = max(
            thresholds,
            key=lambda t: f1_score(
                y[masks["validation"]],
                validation_prob >= t,
                average="macro",
                zero_division=0,
            ),
        )
        for split in ("validation", "test"):
            prob = classifier.predict_proba(x[masks[split]])[:, 1]
            record[split] = classification_report(
                y[masks[split]],
                (prob >= threshold).astype(int),
                labels=[0, 1],
                output_dict=True,
                zero_division=0,
            )
        record["threshold"] = threshold
        # 이 기준은 제품 승인 기준이 아니라 명백히 낮은 실험 결과 차단용이다.
        usable = (
            record["validation"]["macro avg"]["f1-score"] >= 0.65
            and record["validation"]["1"]["precision"] >= 0.65
        )
        record["status"] = "experimental" if usable else "low_validation"
        filename = hashlib.sha256(error.encode()).hexdigest()[:12] + ".joblib"
        joblib.dump(classifier, folder / f"{exercise}-{filename}")
        record["model_file"] = f"{exercise}-{filename}"
        test_indices = np.where(splits == "test")[0]
        probs = classifier.predict_proba(x[test_indices])[:, 1]
        for index, probability in zip(test_indices, probs):
            predictions[rows[index]["id"]].append(
                dict(
                    error=error,
                    name=name,
                    advice=advice,
                    probability=float(probability),
                    threshold=threshold,
                    detected=bool(probability >= threshold),
                    usable=usable,
                )
            )
        result["errors"][error] = record
    # 보기 쉬운 몇 건만 보여 주되 예측 일치 여부로 고르지 않는다.
    test_rows = [r for r in rows if r["split"] == "test"]
    chosen = []
    seen = set()
    for row in sorted(test_rows, key=lambda r: r["id"]):
        category = "reference" if row["reference"] else row["errors"][0]
        if category in seen:
            continue
        seen.add(category)
        chosen.append(dict(row, predictions=predictions[row["id"]]))
    result["demos"] = chosen
    result["valid_clips"] = dict(collections.Counter(splits))
    return result


def main() -> None:
    """전체 자료를 조사하고 모델별 추출·학습을 순서대로 실행한다.

    Raises:
        OSError: 원본 디스크·모델·결과 파일에 접근할 수 없는 경우.
        RuntimeError: GPU 추론 실패. 다른 장치로 자동 대체하지 않는다.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    root = catalog.ROOT
    inventory = catalog.build_catalog()
    save_json(root / "catalog.json", inventory)
    items = inventory["items"]
    LOGGER.info("Selected clips: %s", len(items))
    report_path = root / "results.json"
    results = (
        json.loads(report_path.read_text()) if report_path.exists() else {}
    )
    started = time.time()
    for model in ("heavy",):
        folder = root / "artifacts_v2" / model
        (folder / "features").mkdir(parents=True, exist_ok=True)
        results.setdefault(model, {})
        with pose_features.make_detector(model) as detector:
            done = 0
            for exercise in catalog.PROFILES:
                subset = [r for r in items if r["exercise"] == exercise]
                for row in subset:
                    path = folder / "features" / f"{row['id']}.npz"
                    if not path.exists():
                        x, sequence, coverage = pose_features.extract_clip(
                            detector, Path(row["path"])
                        )
                        temp = path.with_suffix(".pending.npz")
                        np.savez_compressed(
                            temp,
                            X=[] if x is None else x,
                            sequence=sequence,
                            coverage=coverage,
                        )
                        temp.replace(path)
                    done += 1
                    if done % 100 == 0:
                        save_json(
                            root / "progress.json",
                            dict(
                                stage="extract",
                                model=model,
                                exercise=exercise,
                                done=done,
                                total=len(items),
                                elapsed_seconds=time.time() - started,
                            ),
                        )
                        LOGGER.info("%s %s/%s", model, done, len(items))
                results[model][exercise] = fit_exercise(
                    model, exercise, items, folder
                )
                save_json(report_path, results)
                LOGGER.info("%s %s training complete", model, exercise)
        LOGGER.info("%s training complete", model)
    save_json(
        root / "progress.json",
        dict(stage="complete", elapsed_seconds=time.time() - started),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        LOGGER.exception("학습 실행 실패")
        save_json(
            catalog.ROOT / "progress.json",
            dict(
                stage="failed",
                error_type=type(error).__name__,
                message=str(error),
            ),
        )
        raise
