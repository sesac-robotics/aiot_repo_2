"""새 W 시점에서 팔꿈치가 몸통 아래로 붙는 모양을 구분하는 규칙 후보다."""

import itertools
import json
import pathlib

import numpy as np
from sklearn.model_selection import GroupKFold

import audit
from recovery_balance import weights_for
from recovery_prepare import ROOT
from recovery_train import component_groups
from recovery_w_calibration import DEFAULT, NAMES


def classify(features: np.ndarray, policies: np.ndarray) -> np.ndarray:
    """기존 W 조건에 팔꿈치 아래 방향 경계를 더해 등급을 계산한다.

    Args:
        features: N×7 기존 관측 기하. 팔꿈치 상승 비율은 몸통 위 방향과
            상완의 내적이므로 -1에 가까우면 상완이 몸통 아래에 붙는다.
        policies: P×9. 기존 일곱 경계와 정상·부분 수행의 팔꿈치 상승 하한.

    Returns:
        np.ndarray: P×N 정수 등급. 추가 관절·AI 출력·정답 등급을 읽지 않는다.

    Raises:
        ValueError: 배열 형상·유한성·경계 순서 위반.
    """
    if (
        features.ndim != 2
        or features.shape[1] != 7
        or policies.ndim != 2
        or policies.shape[1] != 9
        or not np.isfinite(features).all()
        or not np.isfinite(policies).all()
        or np.any(policies[:, 8] > policies[:, 7])
    ):
        raise ValueError("Expected finite N x 7 and ordered P x 9")
    p = policies[:, :, None]
    arms = []
    for index in (0, 3):
        bend, hand, up = (features[None, :, index + j] for j in range(3))
        arms.append(
            np.minimum.reduce(
                [
                    np.where(
                        bend >= p[:, 0], 2, np.where(bend >= p[:, 1], 1, 0)
                    ),
                    np.where(
                        hand >= p[:, 2], 2, np.where(hand >= p[:, 3], 1, 0)
                    ),
                    np.where(up <= p[:, 4], 2, np.where(up <= p[:, 5], 1, 0)),
                    np.where(up >= p[:, 7], 2, np.where(up >= p[:, 8], 1, 0)),
                ]
            )
        )
    return np.where(
        (np.minimum(*arms) == 2) & (features[None, :, 6] <= p[:, 6]),
        2,
        np.where(np.maximum(*arms) > 0, 1, 0),
    )


def run() -> dict:
    """학습 그룹에서만 하한·손 위치를 선택하고 개발 오판을 대조한다.

    Returns:
        dict: 명시 규칙·그룹 3폴드 및 개발 등급 오차. 개발 사례를 보고
        모양 조건을 설계했으므로 독립 최종 검증이나 전문가 정확도가 아니다.

    Raises:
        OSError: 원자료·결과 접근 실패.
        ValueError: 원자료 지문·그룹 계약 위반.
    """
    phase = audit.read_json(ROOT / "W_phase_trial.json")
    if phase["comparison_sha256"] != audit.digest(ROOT / "comparison.json"):
        raise ValueError("Phase experiment is stale")
    identities = {
        r["id"]: r for r in audit.read_json(ROOT / "fit_input_manifest.json")
    }
    rows = []
    for r in phase["records"]:
        row = dict(r)
        row.update(
            levels=[r["level"]], sources=set(identities[r["id"]]["sources"])
        )
        rows.append(row)
    component_groups(rows)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "validation"]
    groups = component_groups(train)
    policies = []
    for hand, good, partial in itertools.product(
        (0, 0.025, 0.05, 0.1),
        (-1, -0.95, -0.9, -0.85, -0.8),
        (-1, -0.98, -0.95, -0.9),
    ):
        if partial > good:
            continue
        p = list(DEFAULT) + [good, partial]
        p[2] = hand
        policies.append(p)
    policies = np.asarray(policies)
    x = np.asarray([r["feature"] for r in train])
    y = np.asarray([r["level"] for r in train])
    guesses = classify(x, policies)
    distances = np.sum(
        abs(policies - np.asarray(list(DEFAULT) + [-1, -1]))
        / np.asarray([20, 8, 0.1, 0.1, 0.5, 0.5, 35, 0.2, 0.1]),
        axis=1,
    )

    def choose(indices):
        losses = np.average(
            abs(guesses[:, indices] - y[None, indices]),
            axis=1,
            weights=weights_for([train[int(i)] for i in indices], 0, 3),
        )
        return min(
            range(len(policies)), key=lambda i: (losses[i], distances[i], i)
        )

    oof = np.full(len(train), -1)
    for fitting, held in GroupKFold(3).split(train, groups=groups):
        oof[held] = guesses[choose(fitting), held]
    selected = choose(np.arange(len(train)))
    pred = classify(
        np.asarray([r["feature"] for r in val]),
        policies[selected : selected + 1],
    )[0]
    names = list(NAMES) + [
        "elbow_up_good_lower_ratio",
        "elbow_up_partial_lower_ratio",
    ]
    report = dict(
        comparison_sha256=audit.digest(ROOT / "comparison.json"),
        phase_trial_sha256=audit.digest(ROOT / "W_phase_trial.json"),
        code_sha256=audit.digest(pathlib.Path(__file__)),
        candidate_policies=len(policies),
        training_records=len(train),
        training_connected_groups=len(set(groups)),
        selected_policy=dict(zip(names, policies[selected].tolist())),
        group_cv_before_MAE_grade=float(
            np.mean([abs(r["new_rule"] - r["level"]) for r in train])
        ),
        group_cv_after_MAE_grade=float(np.mean(abs(oof - y))),
        fixed_development_records=len(val),
        fixed_before_MAE_grade=float(
            np.mean([abs(r["new_rule"] - r["level"]) for r in val])
        ),
        fixed_after_MAE_grade=float(
            np.mean([abs(int(p) - r["level"]) for r, p in zip(val, pred)])
        ),
        fixed_predictions=[
            dict(
                id=r["id"],
                reference=r["level"],
                before=r["new_rule"],
                after=int(p),
            )
            for r, p in zip(val, pred)
        ],
        development_cases_informed_rule_structure=True,
        validation_targets_used_for_parameter_fitting=False,
        expert_ground_truth=False,
        independent_final_test=False,
        common_rule_promoted=False,
    )
    audit.save_json(ROOT / "W_shape_trial.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
