"""보유 자료의 직접 점수 초안을 관측 승인 후 학습·동일 개발 비교한다.

70개는 최소 한 항목이 실제 학습되는 구간 수이다. 같은 참여자의 인접
유지 구간은 독립 영상·사람으로 계산하지 않으며 총점 가능 수도 분리한다.
"""

import collections
import copy
import json
import pathlib
import sys

import cv2
import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

import audit
import geometry
import plank_item_observation
import recovery_features
import recovery_train
import three_items
import upper_support_event_model
import ytw_projected_rule_trial
from w_observed_absence_trial import detect as observed_w_absence
from allfive_round1_features import load, phase_matches, refined_w
from allfive_round1_inputs import DEFINITIONS
from allfive_round1_train import manifest, predict
from held_feature_update import fit, update_existing, update_note
from held_round2 import OUT
from resources import check_resources


def added_rows(require_review: bool = True) -> list[dict]:
    """원본·직접 초안·관절·추적 승인 교집합을 입력 특징으로 반환한다.

    Args:
        require_review: True이면 직접 관절 검토 승인을 요구한다. False는
            자동 관측 진단 전용이며 학습·학습 확보 수에 사용할 수 없다.

    Returns:
        list[dict]: 등급 0~2 또는 미관측 None과 픽셀 관절 기반 특징 행.
        제공자 문자열·규칙 등급·다른 항목 라벨을 AI 특징으로 넣지 않는다.

    Raises:
        ValueError: 원본·캐시·직접 승인 지문 또는 기존 분할 불일치.
        OSError: 주석·원본·캐시 접근 실패.
    """
    sources = {
        r["id"]: r
        for r in audit.read_json(OUT / "dense_preparation.json")["records"]
    }
    if (OUT / "held_plank_preparation.json").exists():
        sources.update(
            {
                r["id"]: r
                for r in audit.read_json(OUT / "held_plank_preparation.json")[
                    "records"
                ]
            }
        )
    receipts = {
        r["id"]: r for r in audit.read_json(OUT / "extraction.json")["records"]
    }
    reviews = (
        {
            r["id"]: r
            for r in audit.read_json(OUT / "subject_tracking_review.json")[
                "records"
            ]
        }
        if require_review
        else {}
    )
    rows = []
    for d in audit.read_json(OUT / "manual_decisions.json")["records"]:
        source, receipt = sources[d["id"]], receipts[d["id"]]["extraction"]
        if audit.digest(pathlib.Path(source["path"])) != source["sha256"]:
            raise ValueError("Original held video changed")
        segment = receipt["config"]["segment"]
        if [segment["start_frame"], segment["end_frame"]] != d["frames"]:
            raise ValueError("Annotation and observation intervals differ")
        review = reviews.get(d["id"], {})
        if require_review and (
            not review.get("direct_temporal_sample_review")
            or review["fingerprint"] != receipt["fingerprint"]
        ):
            raise ValueError(
                "Tracking was not directly approved for this cache"
            )
        with np.load(receipt["cache"]) as data:
            if str(data["fingerprint"]) != receipt["fingerprint"]:
                raise ValueError("Cache fingerprint differs")
            seq, frames = data["image_xy"].copy(), data["frame_numbers"].copy()
        for first, last in review.get("unreliable_frame_intervals", []):
            seq[(frames >= first) & (frames <= last)] = np.nan
        cap = cv2.VideoCapture(source["path"])
        width, height = cap.get(3), cap.get(4)
        cap.release()
        seq = three_items.mask_image_border(seq, width, height)
        e, levels = source["exercise"], d["levels"]
        ids, weights = DEFINITIONS[e]
        if e == "plank" and any("side" in view for view in d["views"]):
            observation = plank_item_observation.assess(
                seq,
                (d["frames"][1] - d["frames"][0]) / source["fps"],
                [True] * 4,
            )
            channels = geometry.channels(seq)
            notes = []
            for i, item in enumerate(observation["items"]):
                series = channels[
                    item["side"] + "_" + plank_item_observation.CHANNELS[i]
                ]
                valid = series[np.isfinite(series)]
                notes.append(
                    dict(
                        item=ids[i],
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
                        rule_grade=item["rule_grade"],
                        bank="side_video",
                        holds=item["reasons"],
                    )
                )
        elif e == "plank":
            vector, diagnostic = upper_support_event_model.event_features(seq)
            notes = [
                dict(
                    item=item,
                    eligible=i == 3
                    and vector is not None
                    and np.isfinite(vector).all(),
                    feature=(
                        vector.tolist()
                        if i == 3 and vector is not None
                        else None
                    ),
                    measurement=None,
                    higher_is_better=False,
                    rule_grade=None,
                    bank="front_visible_interruption",
                    holds=["partial_front_observation"],
                )
                for i, item in enumerate(ids)
            ]
        elif e == "ytw":
            notes = recovery_features.describe_ytw(seq)
            refined = refined_w(seq)
            if refined:
                notes[2].update(refined, eligible=True)
            for i, note in enumerate(notes[:3]):
                interval = d["phase_windows"].get(note["item"])
                if (
                    i == 2
                    and interval is not None
                    and not phase_matches(note.get("window"), frames, interval)
                ):
                    absence = observed_w_absence(seq)
                    if absence and phase_matches(
                        absence["window"], frames, interval
                    ):
                        note.update(
                            eligible=True,
                            feature=absence["feature"],
                            window=absence["window"],
                            rule_grade=absence["measured_shape_grade"],
                            actual_T_to_rest_observed=True,
                        )
                if levels[i] is not None and (
                    interval is None
                    or not phase_matches(note.get("window"), frames, interval)
                ):
                    note["eligible"] = False
                    note.setdefault("holds", []).append(
                        "phase_label_interval_mismatch"
                    )
        else:
            notes = three_items.describe(e, seq, seq if e == "squats" else None)
            allowed = {
                "pushups": {1, 2},
                "squats": {0, 1, 3},
                "shoulder": {0, 1, 2, 3},
            }[e]
            for i, note in enumerate(notes):
                note["eligible"] = note["eligible"] and i in allowed
        update_note(dict(exercise=e, notes=notes, id=d["id"]), seq)
        for i, note in enumerate(notes):
            note.setdefault("bank", "front")
            note.setdefault("holds", [])
            if levels[i] is None or i in review.get("blocked_item_indices", []):
                note["eligible"] = False
                note["holds"].append(
                    "direct_observation_or_tracking_not_approved"
                )
            feature = note.get("feature")
            if feature is not None and not np.isfinite(feature).all():
                note["eligible"] = False
                note["holds"].append("nonfinite_input_feature")
            note["eligible"] = bool(note["eligible"])
        rows.append(
            dict(
                id=d["id"] + "_held2_score",
                exercise=e,
                person=source["person"],
                source=source["sha256"],
                sources={source["sha256"]},
                split="train",
                recording_groups=[],
                levels=levels,
                notes=notes,
                criterion_ids=ids,
                weights=weights,
                reference_total_allowed=d["attempt_complete"]
                and None not in levels,
                whole_attempt_observed=d["attempt_complete"],
                verified_views=d["views"],
            )
        )
    return rows


def feature_audit() -> dict:
    """추적 승인 전 자동 특징·보류 이유를 저장하되 학습하지 않는다.

    Returns:
        dict: 진단용 구간 수. 이 수는 직접 승인·학습 가능 수가 아니다.

    Raises:
        ValueError: 관측·주석 지문 또는 형상 불일치.
        OSError: 원본·캐시·결과 접근 실패.
    """
    rows = added_rows(False)
    audit.save_json(
        OUT / "automatic_feature_diagnostics.json",
        dict(
            inputs=manifest(rows),
            direct_tracking_review=False,
            training_performed=False,
        ),
    )
    return dict(automatically_diagnosed=len(rows), trained=0)


def run() -> dict:
    """승인된 5종 입력이 각 70구간 이상일 때만 새 후보를 학습·비교한다.

    Returns:
        dict: 항목·구도·등급 분포와 같은 개발 입력의 AI·규칙 비교.
        부족하면 학습을 완료로 표시하지 않고 적격·부족 수를 반환한다.

    Raises:
        ValueError: 기존 역할·원본 중복·관절·직접 승인 계약 위반.
        OSError: 입력·모델·검증 산출물 저장 실패.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    before, after, changes = load()
    # 앞 실험에서 악화된 발 기울기 추가 특징은 보존하고 이번 입력은 기존
    # 스쿼트 특징으로 되돌린다. 기존 개발 결과를 본 설계 변경임을 기록한다.
    rows = [r for r in after if r["exercise"] != "squats"]
    rows += [r for r in before if r["exercise"] == "squats"]
    rows, feature_changes = update_existing(rows)
    audit.save_json(OUT / "feature_changes.json", feature_changes)
    new = added_rows()
    rows += new
    navy_sources = {
        "1f7e9adb5216f0ef0e1cf203ae03fa8242737df2868020d994ee425ad448dd17",
        "e7b52f7bc59927a499bf616d879a8a0dc343b6384f398fe9bca4a200a922331f",
    }
    for row in rows:
        if navy_sources.intersection(row["sources"]):
            row.setdefault("recording_groups", []).append(
                "navy_demonstrations_possible_same_performer_773312_805472"
            )
    recovery_train.component_groups(rows)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate training annotation ID")
    effective = [
        r
        for r in rows
        if r["split"] == "train"
        and any(
            n["eligible"] and g is not None
            for n, g in zip(r["notes"], r["levels"])
        )
    ]
    counts = collections.Counter(r["exercise"] for r in effective)
    qualification = dict(
        effective_training_segments=dict(counts),
        missing_to_70={e: max(0, 70 - counts[e]) for e in DEFINITIONS},
        full_reference_segments={
            e: sum(
                r["exercise"] == e
                and r.get("reference_total_allowed", True)
                and None not in r["levels"]
                and all(n["eligible"] for n in r["notes"])
                for r in effective
            )
            for e in DEFINITIONS
        },
        metadata_groups={
            e: len({r["person"] for r in effective if r["exercise"] == e})
            for e in DEFINITIONS
        },
        counts_are_not_independent_people=True,
        possible_same_navy_performer_conservatively_grouped=True,
        independent_final_test=False,
        external_discovery=False,
    )
    audit.save_json(OUT / "training_qualification.json", qualification)
    audit.save_json(OUT / "candidate_input_manifest.json", manifest(rows))
    if any(qualification["missing_to_70"].values()):
        return dict(qualification, trained=False)
    reports = []
    (OUT / "models").mkdir(exist_ok=True)
    for e in DEFINITIONS:
        check_resources()
        train = [
            r for r in rows if r["exercise"] == e and r["split"] == "train"
        ]
        val = [
            r for r in rows if r["exercise"] == e and r["split"] == "validation"
        ]
        fit_model = (
            (lambda items: ytw_projected_rule_trial.fit(manifest(items)))
            if e == "ytw"
            else fit
        )
        predict_model = (
            (lambda m, r: ytw_projected_rule_trial.predict(m, manifest([r])[0]))
            if e == "ytw"
            else predict
        )
        model = fit_model(train)
        path = OUT / "models" / (e + ".joblib")
        joblib.dump(model, path)
        fixed = [predict_model(model, r) for r in val]
        groups = recovery_train.component_groups(train)
        oof = []
        for fitting, held in GroupKFold(3).split(train, groups=groups):
            check_resources()
            fold = fit_model([train[int(i)] for i in fitting])
            oof.extend(predict_model(fold, train[int(i)]) for i in held)
        heads = (
            [
                dict(
                    item=DEFINITIONS[e][0][i],
                    view="front_projected_manual_observation_gate",
                    records=h["records"],
                    grade_counts=h["grade_counts"],
                    grade_metadata_groups=h["grade_metadata_groups"],
                    policy=h["policy"],
                )
                for i, h in model.items()
            ]
            if e == "ytw"
            else [
                dict(
                    item=k[1],
                    view=k[2],
                    records=h["records"],
                    grade_counts=h["grade_counts"],
                    metadata_groups=h["people"],
                    error_groups=h["error_people"],
                    single_grade=h["single_grade_model"],
                )
                for k, h in model.items()
            ]
        )
        reports.append(
            dict(
                exercise=e,
                training_segments=counts[e],
                fixed=recovery_train.summarize(fixed),
                fixed_predictions=fixed,
                group_cv=recovery_train.summarize(oof),
                group_cv_predictions=oof,
                heads=heads,
                model_sha256=audit.digest(path),
            )
        )
        audit.save_json(
            OUT / "comparison.json", dict(exercises=reports, completed=False)
        )
    report = dict(
        exercises=reports,
        completed=True,
        qualification=qualification,
        independent_final_test=False,
        expert_reference=False,
        promoted=False,
        squat_pitch_rollback_informed_by_existing_development=True,
        ytw_rule_is_projection_proxy=True,
        ytw_automatic_visibility_gate_verified=False,
        external_discovery=False,
    )
    audit.save_json(OUT / "comparison.json", report)
    return dict(trained=True, qualification=qualification)


if __name__ == "__main__":
    print(
        json.dumps(
            feature_audit() if "diagnose" in sys.argv else run(),
            ensure_ascii=False,
        )
    )
