"""독립 직접 등급으로 W의 명시 기하 규칙 경계를 학습 그룹 안에서 보정한다.

회귀기 출력이나 규칙 자체를 정답으로 쓰지 않는다. 공통 규칙은 보존하고
보정 후보의 그룹 검증·고정 개발 검증만 별도 저장한다.
"""

import itertools
import json
import pathlib

import numpy as np
from sklearn.model_selection import GroupKFold

import audit
from recovery_balance import weights_for
from recovery_prepare import ROOT
from recovery_train import component_groups
from resources import check_resources

DEFAULT = (20, 8, 0.05, -0.05, 0.25, 0.5, 35)
NAMES = (
    "bend_good_deg",
    "bend_partial_deg",
    "hand_good_torso_ratio",
    "hand_partial_torso_ratio",
    "elbow_up_good_arm_ratio",
    "elbow_up_partial_arm_ratio",
    "symmetry_good_deg",
)


def classify(features: np.ndarray, policies: np.ndarray) -> np.ndarray:
    """7개 화면 기하 특징에 명시 경계별 W 등급을 계산한다.

    Args:
        features: N×7. 양팔별 굽힘(도)·손 높이(몸통 비율)·팔꿈치 상승
            (상완 비율)과 좌우 굽힘 차이(도). 실제 관측된 값만 허용한다.
        policies: P×7. NAMES 순서의 정상·부분 수행 경계.

    Returns:
        np.ndarray: P×N 정수 0·1·2등급. 양팔 모두 정상과 대칭을 충족하면
        2, 한쪽이라도 부분 수행이면 1, 양팔 모두 미달이면 0이다.

    Raises:
        ValueError: 형상·유한성·정상/부분 경계 순서가 잘못된 경우.
    """
    features, policies = np.asarray(features), np.asarray(policies)
    if (
        features.ndim != 2
        or features.shape[1] != 7
        or policies.ndim != 2
        or policies.shape[1] != 7
        or not np.isfinite(features).all()
        or not np.isfinite(policies).all()
    ):
        raise ValueError("Expected finite N x 7 features and P x 7 policies")
    if (
        np.any(policies[:, 0] <= policies[:, 1])
        or np.any(policies[:, 2] <= policies[:, 3])
        or np.any(policies[:, 4] >= policies[:, 5])
    ):
        raise ValueError("Good/partial boundaries must be ordered")
    p = policies[:, :, None]
    arms = []
    for index in (0, 3):
        bend, hand, elbow = (features[None, :, index + j] for j in range(3))
        arms.append(
            np.minimum.reduce(
                [
                    np.where(
                        bend >= p[:, 0], 2, np.where(bend >= p[:, 1], 1, 0)
                    ),
                    np.where(
                        hand >= p[:, 2], 2, np.where(hand >= p[:, 3], 1, 0)
                    ),
                    np.where(
                        elbow <= p[:, 4], 2, np.where(elbow <= p[:, 5], 1, 0)
                    ),
                ]
            )
        )
    good = (np.minimum(*arms) == 2) & (features[None, :, 6] <= p[:, 6])
    return np.where(good, 2, np.where(np.maximum(*arms) > 0, 1, 0))


def run() -> dict:
    """학습 그룹 3폴드와 기존 개발 영상에서 W 보정 후보를 비교한다.

    Returns:
        dict: 물리 단위가 명시된 선택 경계·등급 MAE·개발 실패 사례.
        공통 규칙 승격·독립 최종 시험·전문가 정확도를 뜻하지 않는다.

    Raises:
        OSError: 학습 기록·비교·결과 저장 실패.
        ValueError: W 특징·기존 규칙 재현·분할 계약 불일치.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    check_resources()
    records = []
    for row in audit.read_json(ROOT / "fit_input_manifest.json"):
        if row["exercise"] != "ytw":
            continue
        item = row["items"][2]
        if (
            row["levels"][2] is None
            or not item["eligible"]
            or item["feature"] is None
        ):
            continue
        records.append(
            dict(
                id=row["id"],
                person=row["person_metadata_group"],
                sources=set(row["sources"]),
                recording_groups=row["recording_groups"],
                split=row["split"],
                levels=[row["levels"][2]],
                feature=item["feature"],
            )
        )
    component_groups(records)
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "validation"]
    groups = component_groups(train)
    candidates = [
        p
        for p in itertools.product(
            (12, 20, 28),
            (4, 8, 12),
            (0, 0.05, 0.1),
            (-0.1, -0.05, 0),
            (0.25, 0.4, 0.55),
            (0.5, 0.65, 0.8),
            (25, 35, 50),
        )
        if p[0] > p[1] and p[2] > p[3] and p[4] < p[5]
    ]
    policies = np.asarray(candidates)
    default = np.asarray([DEFAULT])
    distances = np.sum(
        abs(policies - np.asarray(DEFAULT))
        / np.asarray([20, 8, 0.1, 0.1, 0.5, 0.5, 35]),
        axis=1,
    )
    x = np.asarray([r["feature"] for r in train])
    y = np.asarray([r["levels"][0] for r in train])
    predictions = classify(x, policies)
    original = classify(x, default)[0]

    def choose(indices):
        weights = weights_for([train[int(i)] for i in indices], 0, 3)
        losses = np.average(
            abs(predictions[:, indices] - y[None, indices]),
            axis=1,
            weights=weights,
        )
        return min(
            range(len(policies)), key=lambda i: (losses[i], distances[i], i)
        )

    oof = np.full(len(train), -1)
    fold_policies = []
    for fitting, held in GroupKFold(3).split(train, groups=groups):
        index = choose(fitting)
        oof[held] = predictions[index, held]
        fold_policies.append(dict(zip(NAMES, policies[index].tolist())))
    selected = choose(np.arange(len(train)))
    comparison = audit.read_json(ROOT / "comparison.json")
    reference = next(
        r for r in comparison["exercises"] if r["exercise"] == "ytw"
    )
    fixed = {r["id"]: r for r in reference["fixed_predictions"]}
    val_features = np.asarray([r["feature"] for r in val])
    before = classify(val_features, default)[0] if val else []
    after = (
        classify(val_features, policies[selected : selected + 1])[0]
        if val
        else []
    )
    for row, grade in zip(val, before):
        if fixed[row["id"]]["rule_levels"][2] != int(grade):
            raise ValueError("Default morphology is not reproduced")
    report = dict(
        code_sha256=audit.digest(pathlib.Path(__file__)),
        fit_input_manifest_sha256=audit.digest(
            ROOT / "fit_input_manifest.json"
        ),
        comparison_sha256=audit.digest(ROOT / "comparison.json"),
        training_records=len(train),
        training_connected_groups=len(set(groups)),
        candidate_policies=len(policies),
        selected_policy=dict(zip(NAMES, policies[selected].tolist())),
        original_policy=dict(zip(NAMES, DEFAULT)),
        fold_selected_policies=fold_policies,
        group_cv_records=len(train),
        group_cv_default_MAE_grade=float(np.mean(abs(original - y))),
        group_cv_calibrated_MAE_grade=float(np.mean(abs(oof - y))),
        fixed_development_records=len(val),
        fixed_development_default_MAE_grade=(
            float(
                np.mean(
                    [abs(int(p) - r["levels"][0]) for r, p in zip(val, before)]
                )
            )
            if val
            else None
        ),
        fixed_development_calibrated_MAE_grade=(
            float(
                np.mean(
                    [abs(int(p) - r["levels"][0]) for r, p in zip(val, after)]
                )
            )
            if val
            else None
        ),
        fixed_predictions=[
            dict(
                id=r["id"],
                reference=r["levels"][0],
                default=int(a),
                calibrated=int(b),
                feature=r["feature"],
            )
            for r, a, b in zip(val, before, after)
        ],
        calibration_targets="assistant_direct_temporal_review_nonexpert_drafts",
        validation_targets_used_for_calibration=False,
        rule_outputs_used_as_targets=False,
        expert_ground_truth=False,
        independent_final_test=False,
        common_rule_promoted=False,
        total_ytw_score_validated=False,
    )
    audit.save_json(ROOT / "W_calibration_trial.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
