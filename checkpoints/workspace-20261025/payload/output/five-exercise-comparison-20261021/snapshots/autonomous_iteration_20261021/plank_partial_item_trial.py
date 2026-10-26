"""부분 플랭크 초안을 버리지 않고 관측 항목별 AI·규칙을 비교한다.

기존 총점 모델과 다른 실험이다. 각 AI는 해당 항목의 필수 관절 특징만
입력받으며 다른 항목 라벨·규칙 출력·제공자 문자열을 입력받지 않는다.
"""

import argparse
import pathlib
import collections

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import audit
import benchmark
from plank_item_observation import ITEMS, WEIGHTS, assess, features
from resources import check_resources


def load_rows(enrichment: bool = False) -> list[dict]:
    """기존 학습·부분 초안과 학습에서 제외한 신규 개발 검증을 읽는다.

    Args:
        enrichment: True이면 별도 팔꿈치·종료 사건 학습 초안을 추가한다.

    Returns:
        list[dict]: 관절·등급·출처·참여자·분할·항목 관측 결과 목록.
        기존 부분 초안은 학습 분할만 사용한다. 새 검증은 학습하지 않는다.

    Raises:
        OSError: 라벨·관절 캐시 접근 실패.
        ValueError: 기존 캐시 계약 불일치.
    """
    rows = []
    ids = {
        r["id"]
        for r in audit.read_json(audit.OUT / "comparison.json")["results"][
            "plank"
        ]["rule_only_previews"]
    }
    labels = {r["id"]: r for r in audit.read_json(audit.OUT / "labels.json")}
    for row in audit.read_json(audit.OUT / "development_manifest.json"):
        if row["id"] not in ids:
            continue
        seq, _, _ = benchmark.load_input(row)
        label = labels[row["id"]]
        rows.append(
            dict(
                id=row["id"],
                seq=seq,
                grades=label["criterion_levels"],
                person=row["person"],
                source=row.get("source_original_sha256", row["video_sha256"]),
                split="train",
                duration=(
                    row["segment"]["end_frame"]
                    - row["segment"]["start_frame"]
                    + 1
                )
                / row["segment"]["fps"],
                partial=False,
            )
        )
    direct = audit.read_json(audit.OUT / "plank_error_direct_labels.json")
    revision_path = audit.OUT / "plank_partial_label_revisions.json"
    revision_report = audit.read_json(revision_path)
    if revision_report["base_direct_sha256"] != audit.digest(
        audit.OUT / "plank_error_direct_labels.json"
    ):
        raise ValueError("Partial direct draft revision base changed")
    revisions = {r["id"]: r for r in revision_report["revisions"]}
    for row in direct["reviews"]:
        row = dict(row, draft_levels=list(row["draft_levels"]))
        if row["id"] in revisions:
            revision = revisions[row["id"]]
            index = revision["criterion_index"]
            if row["draft_levels"][index] != revision["previous_grade"]:
                raise ValueError("Revision no longer matches direct draft")
            row["draft_levels"][index] = revision["revised_grade"]
        with np.load(
            audit.OUT / "plank_error_cache" / row["id"] / "primary.npz"
        ) as cache:
            seq = cache["image_xy"].copy()
        rows.append(
            dict(
                id=row["id"],
                seq=seq,
                grades=row["draft_levels"],
                person="youtube_channel_" + row["person_group"],
                source=row["video_sha256"],
                split=row["source_split"],
                duration=(
                    row["segment"]["end_frame"]
                    - row["segment"]["start_frame"]
                    + 1
                )
                / row["segment"]["fps"],
                partial=True,
            )
        )
    for stem in ["plank_balance_challenge", "plank_prospective"]:
        path = audit.OUT / (stem + "_results.json")
        if not path.exists():
            continue
        direct = {
            r["id"]: r
            for r in audit.read_json(audit.OUT / (stem + "_direct.json"))[
                "reviews"
            ]
        }
        exports = {
            r["id"]: r
            for r in audit.read_json(audit.OUT / (stem + "_plan_exports.json"))[
                "records"
            ]
        }
        for row in audit.read_json(path)["results"]:
            with np.load(row["extraction"]["cache"]) as cache:
                seq = cache["image_xy"].copy()
            draft, export = direct[row["id"]], exports[row["id"]]
            rows.append(
                dict(
                    id=row["id"],
                    seq=seq,
                    grades=draft["draft_levels"],
                    person="youtube_channel_" + draft["person_group"],
                    source=draft["source_sha256"],
                    split="validation",
                    duration=export["frames"] / export["fps"],
                    partial=draft["total_score"] is None,
                )
            )
    if enrichment:
        extra_path = audit.OUT / "plank_support_enrichment_direct.json"
        extra = {r["id"]: r for r in audit.read_json(extra_path)["reviews"]}
        exports = {
            r["id"]: r
            for r in audit.read_json(
                audit.OUT / "plank_support_enrichment_plan_exports.json"
            )["records"]
        }
        extraction_report = audit.read_json(
            audit.OUT / "plank_support_extraction.json"
        )
        if audit.digest(extra_path) != extraction_report["direct_sha256"]:
            raise ValueError("Support drafts changed after extraction")
        for entry in extraction_report["records"]:
            draft, export = extra[entry["id"]], exports[entry["id"]]
            if draft["split"] != "train":
                raise ValueError(
                    "Support enrichment only accepts training sources"
                )
            with np.load(entry["extraction"]["cache"]) as cache:
                seq = cache["image_xy"].copy()
            rows.append(
                dict(
                    id=entry["id"],
                    seq=seq,
                    grades=draft["draft_levels"],
                    person="youtube_channel_" + draft["person_group"],
                    source=draft["source_sha256"],
                    split="train",
                    duration=export["frames"] / export["fps"],
                    partial=True,
                    kind=draft["kind"],
                )
            )
    for row in rows:
        row["observation"] = assess(
            row["seq"], row["duration"], [g is not None for g in row["grades"]]
        )
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    if {r["person"] for r in train} & {r["person"] for r in validation} or {
        r["source"] for r in train
    } & {r["source"] for r in validation}:
        raise ValueError("Training/validation source or person leakage")
    return rows


def fit(rows: list[dict], index: int):
    """학습 폴드 안에서 등급·참여자 균형과 결측 대체값을 계산한다.

    Args:
        rows: 해당 항목 관측을 통과한 학습 자료 목록.
        index: 0~3 항목 인덱스.

    Returns:
        tuple: fitted sklearn Pipeline과 입력 특징 이름 목록.
        모델 출력은 0~2 등급의 실수 예측이다.

    Raises:
        ValueError: 학습 자료가 없거나 특징 차원이 맞지 않는 경우.
    """
    if not rows:
        raise ValueError("No eligible fitting rows")
    x = [
        features(r["seq"], index, r["observation"]["items"][index]["side"])[0]
        for r in rows
    ]
    y = [r["grades"][index] for r in rows]
    pairs = collections.Counter(
        (g, r["person"]) for g, r in zip(y, rows, strict=True)
    )
    groups = {g: len({p for v, p in pairs if v == g}) for g in set(y)}
    weights = np.asarray(
        [
            1 / (len(groups) * groups[g] * pairs[g, r["person"]])
            for g, r in zip(y, rows, strict=True)
        ]
    )
    weights /= weights.mean()
    model = make_pipeline(
        SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        ),
        ExtraTreesRegressor(
            n_estimators=128, min_samples_leaf=2, random_state=21021, n_jobs=1
        ),
    )
    model.fit(
        np.asarray(x), np.asarray(y), extratreesregressor__sample_weight=weights
    )
    return (
        model,
        features(
            rows[0]["seq"],
            index,
            rows[0]["observation"]["items"][index]["side"],
        )[1],
    )


def run(enrichment: bool = False) -> dict:
    """항목별 학습·참여자 제외 진단·동결 개발 검증 결과를 저장한다.

    Args:
        enrichment: True이면 추가 초안으로 별도 support 모델·보고서를 만든다.

    Returns:
        dict: 항목별 유효 수·부분 자료 추가 수·등급 오차·검증 예측.
        총점 모델을 교체하지 않으며 독립 최종 시험으로 표현하지 않는다.

    Raises:
        OSError: 입력·출력·모델 저장 실패.
        ValueError: 관측 계약 또는 원본·참여자 분할 누출.
        RuntimeError: 자원 보호 중단.
    """
    check_resources()
    data = load_rows(enrichment)
    stem = (
        "plank_support_item_trial" if enrichment else "plank_partial_item_trial"
    )
    output = audit.OUT / stem
    output.mkdir(exist_ok=True)
    reports = []
    for index, item in enumerate(ITEMS):
        fitting = [
            r
            for r in data
            if r["split"] == "train"
            and r["observation"]["items"][index]["eligible"]
        ]
        testing = [
            r
            for r in data
            if r["split"] == "validation"
            and r["observation"]["items"][index]["eligible"]
        ]
        predictions = []
        for held in sorted({r["person"] for r in fitting}):
            test = [r for r in fitting if r["person"] == held]
            sources = {r["source"] for r in test}
            train = [
                r
                for r in fitting
                if r["person"] != held and r["source"] not in sources
            ]
            if not train:
                continue
            model, _ = fit(train, index)
            for row in test:
                x, _ = features(
                    row["seq"],
                    index,
                    row["observation"]["items"][index]["side"],
                )
                predictions.append(
                    dict(
                        id=row["id"],
                        person=held,
                        draft_grade=row["grades"][index],
                        ai_grade=float(np.clip(model.predict([x])[0], 0, 2)),
                        rule_grade=row["observation"]["items"][index][
                            "rule_grade"
                        ],
                    )
                )
        model, names = fit(fitting, index)
        joblib.dump(
            dict(
                model=model,
                names=names,
                criterion_id=item,
                required_joints="item_observation_v1",
                train_ids=[r["id"] for r in fitting],
                expert_validated=False,
            ),
            output / (item + ".joblib"),
        )
        validation = []
        for row in testing:
            x, _ = features(
                row["seq"], index, row["observation"]["items"][index]["side"]
            )
            validation.append(
                dict(
                    id=row["id"],
                    draft_grade=row["grades"][index],
                    ai_grade=float(np.clip(model.predict([x])[0], 0, 2)),
                    rule_grade=row["observation"]["items"][index]["rule_grade"],
                )
            )
        reports.append(
            dict(
                criterion_id=item,
                model_status=(
                    "single_grade_constant_baseline"
                    if len({r["grades"][index] for r in fitting}) < 2
                    else "unvalidated_item_regressor"
                ),
                weight=WEIGHTS[index],
                training_records=len(fitting),
                training_people=len({r["person"] for r in fitting}),
                training_nonfull_grade_people=len(
                    {r["person"] for r in fitting if r["grades"][index] < 2}
                ),
                training_grade_counts=dict(
                    collections.Counter(
                        str(r["grades"][index]) for r in fitting
                    )
                ),
                added_partial_records=[
                    r["id"] for r in fitting if r["partial"]
                ],
                person_heldout=predictions,
                person_heldout_ai_MAE_grade=(
                    float(
                        np.mean(
                            [
                                abs(r["ai_grade"] - r["draft_grade"])
                                for r in predictions
                            ]
                        )
                    )
                    if predictions
                    else None
                ),
                person_heldout_rule_MAE_grade=(
                    float(
                        np.mean(
                            [
                                abs(r["rule_grade"] - r["draft_grade"])
                                for r in predictions
                            ]
                        )
                    )
                    if predictions
                    else None
                ),
                validation=validation,
            )
        )
    report = dict(
        items=reports,
        records=[{k: v for k, v in r.items() if k != "seq"} for r in data],
        common_model_modified=False,
        common_rule_modified=False,
        labels_modified=False,
        expert_validated=False,
        independent_final_test=False,
        total_scores_not_renormalized=True,
        version="item-joint-balanced-development-v1",
        dataset_variant=(
            "support_enrichment" if enrichment else "partial_items_base"
        ),
        support_direct_sha256=(
            audit.digest(audit.OUT / "plank_support_enrichment_direct.json")
            if enrichment
            else None
        ),
        direct_revision_file=str(
            audit.OUT / "plank_partial_label_revisions.json"
        ),
        revision_after_model_outputs_seen=True,
    )
    audit.save_json(audit.OUT / (stem + ".json"), report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-enrichment", action="store_true")
    result = run(parser.parse_args().support_enrichment)
    print(
        [
            (
                r["criterion_id"],
                r["training_records"],
                r["training_grade_counts"],
                r["validation"],
            )
            for r in result["items"]
        ]
    )
