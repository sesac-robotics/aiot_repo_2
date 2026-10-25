"""두 번째 보유 스쿼트 검토의 관절 입력·분할·재학습을 별도로 검증한다."""

import importlib.util
import collections
import json
import pathlib
import types

import joblib
import cv2
import numpy as np
from sklearn.model_selection import GroupKFold

import audit
import extract
import recovery_features
import recovery_train
import three_items
import three_training
from multiview_squat_batch2 import OUT, PREVIOUS
from multiview_squat_trial import validate_alignment
from recovery_prepare import ROOT
from resources import check_resources


def mask_reviewed_frames(
    sequence: np.ndarray, frames: np.ndarray, intervals: list[list[int]]
) -> np.ndarray:
    """직접 표본 대조에서 불확실하다고 기록한 원본 프레임만 제외한다.

    Args:
        sequence: N×33×2 원본 픽셀 좌표. 입력 사본을 사용한다.
        frames: N개 원본 프레임 번호.
        intervals: 제외할 원본 프레임의 포함 시작·끝 번호 목록.

    Returns:
        np.ndarray: 해당 프레임 좌표만 NaN인 사본. 보간·자동 사람 판정은
        하지 않으며 다른 프레임의 실제 관측 정보는 보존한다.

    Raises:
        ValueError: 좌표·시각 형상 또는 구간 순서가 유효하지 않은 경우.
    """
    if sequence.shape != (len(frames), 33, 2) or frames.ndim != 1:
        raise ValueError("Expected N x 33 x 2 and N source frame numbers")
    result = sequence.copy()
    for first, last in intervals:
        if first > last:
            raise ValueError("Invalid rejected frame interval")
        result[(frames >= first) & (frames <= last)] = np.nan
    return result


def fit_models(rows: list[dict]) -> dict:
    """학습 역할만으로 AI와 무릎·뒤꿈치의 네 규칙 경계를 보정한다.

    Args:
        rows: train 역할의 직접 등급·원본·관측 특징 행.

    Returns:
        dict: 기존 AI 구현과 항목별 규칙. 기존 3종 학습기의 뒤꿈치 경계
        후보를 재사용하며 검증 라벨·제공자 문자열은 선택에 쓰지 않는다.
        관측과 오류 참여자 메타 그룹이 부족하면 기존 경계를 유지한다.

    Raises:
        ValueError: 검증 역할 입력 또는 규칙·특징 계약 불일치.
    """
    if any(r["split"] != "train" for r in rows):
        raise ValueError("Only training roles can calibrate a model")
    models = recovery_train.fit(rows)
    for key, head in models.items():
        if key[1] != "knee_foot":
            continue
        bucket = []
        for row in rows:
            for i, note in enumerate(row["notes"]):
                grade = row["levels"][i]
                if (
                    recovery_train.head_key(row, i) == key
                    and grade is not None
                    and note["eligible"]
                    and note["measurement"] is not None
                    and np.isfinite(note["feature"]).all()
                ):
                    bucket.append((row, note, grade))
        counts = collections.Counter((g, r["person"]) for r, n, g in bucket)
        grade_people = collections.Counter(g for g, p in counts)
        if (
            len({r["person"] for r, n, g in bucket if g < 2}) < 2
            or len(grade_people) < 2
            or min(grade_people.values()) < 2
        ):
            continue
        weights = [
            1 / (grade_people[g] * counts[g, r["person"]]) for r, n, g in bucket
        ]
        prior = head["boundary"] or three_training.DEFAULTS["knee_foot"]
        prior = tuple(prior) if len(prior) == 4 else (*prior, 0.025, 0.08)
        candidates = [
            (g, p, hg, hp)
            for g in three_training.BOUNDARIES["knee_foot"][0]
            for p in three_training.BOUNDARIES["knee_foot"][1]
            if g < p
            for hg in (0.01, 0.025, 0.04, 0.06)
            for hp in (0.04, 0.06, 0.08, 0.12, 0.16)
            if hg < hp
        ]
        candidates.append(prior)

        def loss(bounds):
            return float(
                np.average(
                    [
                        abs(three_training.rule_grade(n, bounds) - g)
                        for r, n, g in bucket
                    ],
                    weights=weights,
                )
            )

        chosen = min(
            candidates,
            key=lambda b: (
                loss(b),
                sum(
                    abs(a - c) / s
                    for a, c, s in zip(b, prior, (0.25, 0.35, 0.05, 0.12))
                ),
            ),
        )
        head["boundary"] = chosen
        head["heel_calibration"] = dict(
            previous_boundary=prior,
            selected_boundary=chosen,
            training_weighted_MAE_before=loss(prior),
            training_weighted_MAE_after=loss(chosen),
            candidate_policies=len(candidates),
            training_only=True,
            numeric_grid_reused_from_existing_three_training=True,
        )
    return models


def configured_script(filename: str, folder: pathlib.Path) -> types.ModuleType:
    """동일 구현을 독립 모듈로 읽어 이번 출력 폴더만 설정한다.

    Args:
        filename: 이 실험 폴더의 기존 Python 파일 이름.
        folder: 라벨·관절·검토 기록이 있는 절대 출력 폴더.

    Returns:
        types.ModuleType: 별도 이름 공간의 구현. 기존 모듈·파일은 변경하지
        않는다. 실행 중 경로 설정을 출력 지문에 별도로 기록한다.

    Raises:
        ImportError: 모듈을 읽거나 초기화하지 못한 경우.
        ValueError: 외부 폴더 파일 이름이 입력된 경우.
    """
    if pathlib.Path(filename).name != filename:
        raise ValueError("Only experiment-local implementations are allowed")
    path = pathlib.Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(
        "batch2_isolated_" + path.stem, path
    )
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = folder
    return module


def extract_rows() -> dict:
    """직접 승인한 구간만 두 시점 Heavy로 추출하고 원본 시차를 확인한다.

    Returns:
        dict: 시점별 모델·원본·구간·캐시 지문. 직접 검토·점수 부여는 아니다.

    Raises:
        ValueError: 기존 개발 역할·라벨·원본 시차 계약 불일치.
        RuntimeError: 온도·메모리 보호 기준 위반.
        OSError: 라벨·원본·캐시 저장 실패.
    """
    records = []
    for row in audit.read_json(OUT / "score_annotations.json"):
        check_resources()
        if row["split"] not in ("train", "validation"):
            raise ValueError("Reserved test cannot enter batch2")
        primary = dict(row, required_views=["side", "synchronized_front"])
        side = extract.extract_view(
            primary, "primary", row["video_path"], OUT / "heavy"
        )
        front = extract.extract_view(
            dict(primary, segment=row["front_segment"]),
            "front",
            row["front_path"],
            OUT / "heavy",
        )
        with np.load(side["cache"]) as s, np.load(front["cache"]) as f:
            validate_alignment(
                row["alignment"], s["frame_numbers"], f["frame_numbers"]
            )
        records.append(dict(id=row["id"], side=side, front=front))
        audit.save_json(
            OUT / "extraction.json", dict(records=records, completed=False)
        )
    result = dict(records=records, completed=True)
    audit.save_json(OUT / "extraction.json", result)
    return dict(attempts=len(records), views=len(records) * 2)


def load_batch(folder: pathlib.Path) -> list[dict]:
    """기존과 같은 관측 구현으로 직접 관절 검토와 각 항목의 입력을 대조한다.

    Args:
        folder: 해당 배치의 라벨·추출·직접 추적 검토 기록 폴더.

    Returns:
        list[dict]: 같은 특징·관측 정책을 적용한 학습 입력. 불완전 반복의
        총점은 허용하지 않으며 보이는 항목은 유지한다.

    Raises:
        ValueError: 직접 검토·입력·캐시·분할 계약 불일치.
        OSError: 파일 접근 실패.
    """
    module = configured_script("multiview_squat_trial.py", folder)
    receipts = audit.read_json(folder / "extraction.json")["records"]
    rows = module.load_new(receipts)
    annotations = {
        r["id"]: r for r in audit.read_json(folder / "score_annotations.json")
    }
    tracking = {
        (r["id"], r["view"]): r
        for r in audit.read_json(folder / "subject_tracking_review.json")[
            "records"
        ]
    }
    receipt_index = {r["id"]: r for r in receipts}
    for row in rows:
        ann = annotations[row["id"]]
        if any(
            tracking[row["id"], view].get("unreliable_frame_intervals")
            for view in ("side", "front")
        ):
            sequences = []
            for view, path in (
                ("side", ann["video_path"]),
                ("front", ann["front_path"]),
            ):
                with np.load(receipt_index[row["id"]][view]["cache"]) as data:
                    seq = mask_reviewed_frames(
                        data["image_xy"],
                        data["frame_numbers"],
                        tracking[row["id"], view].get(
                            "unreliable_frame_intervals", []
                        ),
                    )
                cap = cv2.VideoCapture(path)
                try:
                    seq = three_items.mask_image_border(
                        seq, cap.get(3), cap.get(4)
                    )
                finally:
                    cap.release()
                sequences.append(seq)
            row["notes"] = three_items.describe("squats", *sequences)
            for i, note in enumerate(row["notes"]):
                note["bank"] = "front" if i == 3 else "side"
            for view in ("side", "front"):
                for i in tracking[row["id"], view].get(
                    "blocked_item_indices", []
                ):
                    row["notes"][i]["eligible"] = False
                    row["notes"][i]["holds"].append(
                        "direct_sample_review_unreliable_target_geometry"
                    )
        for note in row["notes"]:
            if note["item"] in ann.get("manual_unobservable_items", []):
                note["eligible"] = False
                note["holds"].append("direct_video_review_item_not_observable")
        row["reference_total_allowed"] = ann.get(
            "eligible_for_total_score_training", True
        )
        row["whole_attempt_observed"] = ann.get("attempt_complete", True)
    return rows


def screen() -> dict:
    """직접 추적 승인 전 항목별 자동 관측 보류를 진단용으로 집계한다.

    Returns:
        dict: 자동 판정 가능한 구간·항목 후보 수. 직접 검토·동일인 승인·
        학습 허가는 생성하지 않으며 수치가 좋아도 직접 추적 대조가 필요하다.

    Raises:
        ValueError: 캐시 좌표 형상·원본 크기 또는 시점 길이 불일치.
        OSError: 라벨·캐시·진단 저장 실패.
        RuntimeError: 자원 보호 기준 위반.
    """
    annotations = {
        r["id"]: r for r in audit.read_json(OUT / "score_annotations.json")
    }
    records = []
    for receipt in audit.read_json(OUT / "extraction.json")["records"]:
        check_resources()
        row = annotations[receipt["id"]]
        sequences = []
        for view, path in (
            ("side", row["video_path"]),
            ("front", row["front_path"]),
        ):
            with np.load(receipt[view]["cache"]) as data:
                seq = data["image_xy"].copy()
            cap = cv2.VideoCapture(path)
            try:
                seq = three_items.mask_image_border(seq, cap.get(3), cap.get(4))
            finally:
                cap.release()
            sequences.append(seq)
        notes = three_items.describe("squats", *sequences)
        for note in notes:
            if note["item"] in row.get("manual_unobservable_items", []):
                note["eligible"] = False
                note["holds"].append("direct_video_review_item_not_observable")
        records.append(
            dict(
                id=row["id"],
                split=row["split"],
                automatic_four_item_candidate=all(n["eligible"] for n in notes)
                and row["eligible_for_total_score_training"],
                items=[
                    dict(
                        item=n["item"],
                        eligible=n["eligible"],
                        simultaneous_observation=n["simultaneous_observation"],
                        holds=n["holds"],
                    )
                    for n in notes
                ],
                direct_tracking_approved=False,
            )
        )
    result = dict(records=records, automatic_diagnostic_not_direct_review=True)
    audit.save_json(OUT / "observation_screen.json", result)
    return dict(
        automatic_total_candidates=sum(
            r["automatic_four_item_candidate"] for r in records
        ),
        validation_total_candidates=sum(
            r["automatic_four_item_candidate"] and r["split"] == "validation"
            for r in records
        ),
    )


def prediction(model: dict, row: dict) -> dict:
    """동일 직접 초안과 관측 입력에서 AI·규칙 예측 행을 만든다.

    Args:
        model: 학습 자료만으로 정한 항목별 AI와 규칙 경계.
        row: 직접 등급·배점·기하 특징·관측 정책이 있는 입력 행.

    Returns:
        dict: 0~2 등급·0~100 총점·근거. 누락 항목의 총점은 None이다.

    Raises:
        ValueError: 특징·등급 또는 모델 입력 불일치.
    """
    return dict(
        id=row["id"],
        person=row["person"],
        criterion_ids=row["criterion_ids"],
        levels=row["levels"],
        target_score=(
            three_items.total(row["levels"], row["weights"])
            if row.get("reference_total_allowed", True)
            else None
        ),
        **recovery_train.predict(model, row),
        observation=[
            dict(
                item=n["item"], eligible=n["eligible"], holds=n.get("holds", [])
            )
            for n in row["notes"]
        ]
    )


def fit() -> dict:
    """첫 배치 후보와 새 후보를 같은 개발 검증 입력에서 비교한다.

    Returns:
        dict: 학습·항목 분포·개발 검증·연결 그룹 교차검증 결과. 독립 최종
        시험이나 전문가 정답 정확도를 뜻하지 않으며 자동 승격하지 않는다.

    Raises:
        ValueError: 원본·참여자 누출 또는 보존 모델·입력 변경.
        RuntimeError: 온도·메모리 보호 기준 위반.
        OSError: 입력·모델·비교 결과 저장 실패.
    """
    protected = [
        ROOT / "comparison.json",
        ROOT / "fit_input_manifest.json",
        PREVIOUS / "comparison.json",
        PREVIOUS / "squats_candidate.joblib",
    ] + list((ROOT / "models").glob("*.joblib"))
    preserved = {str(p): audit.digest(p) for p in protected}
    base, extra = recovery_features.load()
    original = [r for r in base + extra if r["exercise"] == "squats"]
    prior = load_batch(PREVIOUS)
    added = load_batch(OUT)
    rows = original + prior + added
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate attempt IDs")
    groups = recovery_train.component_groups(rows)
    train_indices = [i for i, r in enumerate(rows) if r["split"] == "train"]
    training = [rows[i] for i in train_indices]
    model = fit_models(training)
    prior_heel_model = {
        k: dict(
            v,
            boundary=v.get("heel_calibration", {}).get(
                "previous_boundary", v["boundary"]
            ),
        )
        for k, v in model.items()
    }
    baseline = joblib.load(PREVIOUS / "squats_candidate.joblib")
    joblib.dump(model, OUT / "squats_candidate.joblib")
    new_ids = {r["id"] for r in added}
    prior_ids = {r["id"] for r in prior}
    predictions = []
    for row in rows:
        if row["split"] != "validation":
            continue
        predictions.append(
            dict(
                prediction(model, row),
                baseline=prediction(baseline, row),
                before_heel_calibration=prediction(prior_heel_model, row),
                subset=(
                    "batch2"
                    if row["id"] in new_ids
                    else (
                        "batch1"
                        if row["id"] in prior_ids
                        else "existing_development"
                    )
                ),
            )
        )
    comparisons = {}
    rule_ablation = {}
    for subset in ("existing_development", "batch1", "batch2"):
        selected = [r for r in predictions if r["subset"] == subset]
        comparisons[subset] = dict(
            baseline=recovery_train.summarize(
                [p["baseline"] for p in selected]
            ),
            candidate=recovery_train.summarize(selected),
        )
        rule_ablation[subset] = dict(
            prior_heel_boundaries=recovery_train.summarize(
                [p["before_heel_calibration"] for p in selected]
            ),
            calibrated=recovery_train.summarize(selected),
        )
    cross = []
    training_groups = [groups[i] for i in train_indices]
    if len(set(training_groups)) >= 3:
        for fitting, held in GroupKFold(
            min(5, len(set(training_groups)))
        ).split(training, groups=training_groups):
            check_resources()
            fold_model = fit_models([training[int(i)] for i in fitting])
            cross.extend(prediction(fold_model, training[int(i)]) for i in held)
    effective = [
        r
        for r in training
        if any(
            n["eligible"] and g is not None
            for n, g in zip(r["notes"], r["levels"])
        )
    ]
    for path, sha in preserved.items():
        if audit.digest(pathlib.Path(path)) != sha:
            raise ValueError("Preserved result changed")
    features = [
        dict(
            id=r["id"],
            person=r["person"],
            split=r["split"],
            sources=sorted(r["sources"]),
            recording_groups=r.get("recording_groups", []),
            levels=r["levels"],
            reference_total_allowed=r.get("reference_total_allowed", True),
            whole_attempt_observed=r.get("whole_attempt_observed", True),
            items=[
                dict(
                    item=n["item"],
                    bank=n["bank"],
                    eligible=n["eligible"],
                    feature=n["feature"] if n["eligible"] else None,
                )
                for n in r["notes"]
            ],
        )
        for r in rows
    ]
    audit.save_json(OUT / "fit_input_manifest.json", features)
    report = dict(
        completed=True,
        independent_final_test=False,
        expert_reference=False,
        promoted=False,
        baseline="preserved_first_multiview_candidate",
        preserved=preserved,
        model_sha256=audit.digest(OUT / "squats_candidate.joblib"),
        fit_input_manifest_sha256=audit.digest(OUT / "fit_input_manifest.json"),
        configuration=dict(
            loaded_folder=str(OUT),
            prior_folder=str(PREVIOUS),
            library_path_override_in_isolated_module=True,
        ),
        input_sha256={
            name: audit.digest(OUT / name)
            for name in (
                "score_annotations.json",
                "subject_tracking_review.json",
                "extraction.json",
                "manual_decisions.json",
            )
        },
        code_sha256={
            name: audit.digest(pathlib.Path(__file__).with_name(name))
            for name in (
                "multiview_squat_batch2_train.py",
                "multiview_squat_batch2.py",
                "multiview_squat_trial.py",
                "recovery_features.py",
                "recovery_train.py",
                "three_items.py",
                "three_training.py",
                "geometry.py",
            )
        },
        training_effective_segments=len(effective),
        training_four_item_observed_segments=sum(
            None not in r["levels"]
            and all(n["eligible"] for n in r["notes"])
            and r.get("reference_total_allowed", True)
            for r in training
        ),
        training_metadata_groups=len({r["person"] for r in effective}),
        training_connected_groups=len(set(training_groups)),
        added=dict(
            annotations=len(added),
            train_segments=sum(r["split"] == "train" for r in added),
            validation_segments=sum(r["split"] == "validation" for r in added),
            train_eligible_item_labels=sum(
                n["eligible"] and g is not None
                for r in added
                if r["split"] == "train"
                for n, g in zip(r["notes"], r["levels"])
            ),
            validation_eligible_item_labels=sum(
                n["eligible"] and g is not None
                for r in added
                if r["split"] == "validation"
                for n, g in zip(r["notes"], r["levels"])
            ),
            four_item_observed_segments=sum(
                all(n["eligible"] for n in r["notes"])
                and r.get("reference_total_allowed", True)
                for r in added
            ),
        ),
        heads=[
            dict(
                exercise=k[0],
                item=k[1],
                view=k[2],
                records=v["records"],
                metadata_groups=v["people"],
                grade_counts=v["grade_counts"],
                error_metadata_groups=v["error_people"],
                rule_boundary=v["boundary"],
                heel_calibration=v.get("heel_calibration"),
            )
            for k, v in model.items()
        ],
        comparisons=comparisons,
        rule_calibration_ablation=rule_ablation,
        group_cross_validation=recovery_train.summarize(cross),
        predictions=predictions,
        split_leakage=False,
        same_source_camera_views_not_independent_people=True,
    )
    audit.save_json(OUT / "comparison.json", report)
    return {
        k: report[k]
        for k in (
            "training_effective_segments",
            "training_four_item_observed_segments",
            "added",
            "comparisons",
            "group_cross_validation",
        )
    }


if __name__ == "__main__":
    import sys

    stage = sys.argv[1]
    if stage == "extract":
        result = extract_rows()
    elif stage == "extract-and-joints":
        result = dict(
            extraction=extract_rows(),
            automatic_screen=screen(),
            joints=configured_script(
                "multiview_squat_joint_review.py", OUT
            ).run(),
        )
    elif stage == "joints":
        result = configured_script("multiview_squat_joint_review.py", OUT).run()
    elif stage == "fit":
        result = fit()
    elif stage == "screen":
        result = screen()
    else:
        raise ValueError("Unknown batch2 stage")
    print(json.dumps(result, ensure_ascii=False))
