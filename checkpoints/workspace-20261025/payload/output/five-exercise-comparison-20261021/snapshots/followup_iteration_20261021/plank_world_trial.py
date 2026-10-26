"""같은 관측 관절의 Heavy 추정 3D 각도로 투영 왜곡을 진단한다.

world 좌표는 미터 단위 추정값이며 실측 3D 정답이 아니다. 원본 화면에서
미관측인 관절은 계속 제외한다. Y 깊이·바닥 접촉·하중을 복원했다고 주장하지 않는다.
"""

import collections

import joblib
import numpy as np

import audit
import benchmark
import geometry
import rule_score
from cpu_analysis_guard import check_resources
from followup_learning import WEIGHTS, total_metrics
from plank_monotone_trial import fit, predict
from plank_partial_item_trial import load_rows


def measurements(
    image_xy: np.ndarray, world_xyz: np.ndarray
) -> tuple[np.ndarray | None, dict]:
    """보이는 관절의 3D 선·무릎 각도와 2D 팔꿈치 오프셋을 반환한다.

    Args:
        image_xy: N×33×2 원본 픽셀 좌표. NaN은 미관측.
        world_xyz: N×33×3 Heavy 추정 미터 좌표. 같은 표본 순서다.

    Returns:
        tuple: 네 기하 통계 또는 None, 2D·3D 선 각도 및 선택 측면 근거.
        같은 관측 마스크와 공통 측면을 사용한다. 80% 미만 동시 관측은 보류한다.

    Raises:
        ValueError: 배열 형상 또는 표본 순서 길이 불일치.
    """
    image = np.asarray(image_xy, dtype=float)
    world = np.asarray(world_xyz, dtype=float).copy()
    if (
        image.ndim != 3
        or image.shape[1:] != (33, 2)
        or world.shape != (len(image), 33, 3)
    ):
        raise ValueError("Expected matching image/world landmark sequences")
    world[~np.isfinite(image).all(2)] = np.nan
    channels = geometry.channels(image)
    side = rule_score._side(channels)
    offset = int(side == "right")
    joints = np.array([11, 23, 25, 27]) + offset
    valid = np.isfinite(world[:, joints]).all((1, 2))
    note = dict(
        side=side,
        joint_observation_fraction=float(valid.mean()),
        image_line_p10_deg=rule_score._stat(channels[side + "_line_deg"], 10),
        world_line_p10_deg=None,
    )
    if len(image) < 5 or valid.mean() < 0.8:
        return None, note
    shoulder, hip, knee, ankle = (world[:, j] for j in joints)
    line = geometry.angle_deg(shoulder, hip, ankle)
    leg = geometry.angle_deg(hip, knee, ankle)
    line_value = rule_score._stat(line, 10)
    knee_value = rule_score._stat(leg, 10)
    elbow = rule_score._stat(channels[side + "_elbow_offset"], 90)
    finite = leg[np.isfinite(leg)]
    if line_value is None or knee_value is None or elbow is None:
        return None, note
    note["world_line_p10_deg"] = line_value
    return (
        np.asarray(
            [
                line_value,
                elbow,
                knee_value,
                np.percentile(finite, 95) - np.percentile(finite, 5),
            ]
        ),
        note,
    )


def run() -> dict:
    """이미 노출된 같은 12개 개발 구간에서 2D·3D 단조 회귀를 비교한다.

    Returns:
        dict: 같은 참여자·원본 제외 예측과 투영 진단. 기존 새 검증 출처를
        다시 평가하지 않는다. 단일 등급 항목은 상수 기준선임을 표시한다.

    Raises:
        OSError: 캐시·직접 초안·모델 저장 실패.
        ValueError: 기존 비교 대상과 불일치 또는 관측 실패.
        RuntimeError: CPU 분석 자원 보호 실패.
    """
    raws = []
    for row in load_rows(True):
        if row["split"] == "train" and not row["partial"]:
            raws.append(
                dict(row, score=float(np.dot(row["grades"], WEIGHTS) / 2))
            )
    for draft in audit.read_json(audit.OUT / "followup_plank_direct.json")[
        "reviews"
    ]:
        if draft["split"] != "train":
            continue
        seq, _, holds = benchmark.load_input(dict(draft, exercise="plank"))
        if holds:
            continue
        raws.append(
            dict(
                id=draft["id"],
                seq=seq,
                grades=draft["draft_levels"],
                score=draft["total_score"],
                person=draft["person"],
                source=draft["video_sha256"],
            )
        )
    cohort = []
    diagnostics = []
    for row in raws:
        with np.load(audit.OUT / "heavy" / row["id"] / "side.npz") as cache:
            x, note = measurements(row["seq"], cache["world_xyz"])
        diagnostics.append(
            dict(id=row["id"], direct_trunk_grade=row["grades"][0], **note)
        )
        if x is None:
            raise ValueError(
                "World trial observation differs from paired cohort"
            )
        cohort.append(dict(row, x=x))
    baseline = audit.read_json(audit.OUT / "plank_monotone_trial_results.json")
    previous = {r["id"]: r for r in baseline["LOPO_predictions"]}
    if set(previous) != {r["id"] for r in cohort}:
        raise ValueError("2D and 3D cohorts differ")
    predictions = []
    for person in sorted({r["person"] for r in cohort}):
        check_resources()
        testing = [r for r in cohort if r["person"] == person]
        sources = {r["source"] for r in testing}
        models = fit(
            [
                r
                for r in cohort
                if r["person"] != person and r["source"] not in sources
            ]
        )
        for row in testing:
            score, levels, outside = predict(models, row["x"])
            predictions.append(
                dict(
                    id=row["id"],
                    held_person=person,
                    target_score=row["score"],
                    before_ai=previous[row["id"]]["after_ai"],
                    after_ai=score,
                    rule_score=previous[row["id"]]["rule_score"],
                    levels=levels,
                    outside_training_feature_range=outside,
                )
            )
    models = fit(cohort)
    path = audit.OUT / "followup_plank_model/world_monotone.joblib"
    joblib.dump(
        dict(
            models=models,
            train_ids=[r["id"] for r in cohort],
            world_is_estimated=True,
            expert_validated=False,
        ),
        path,
    )
    report = dict(
        LOPO_predictions=predictions,
        metrics=total_metrics(predictions),
        before_ai_meaning="2D_monotone",
        after_ai_meaning="observed_landmark_estimated_3D_monotone",
        projection_diagnostics=diagnostics,
        training_records=len(cohort),
        item_grade_counts=[
            dict(collections.Counter(r["grades"][i] for r in cohort))
            for i in range(4)
        ],
        world_is_estimated_not_measured=True,
        unobserved_joints_used=False,
        rule_targets_used=False,
        validation_evaluation_repeated=False,
        posthoc_development_trial=True,
        independent_final_test=False,
        expert_validated=False,
        common_model_replaced=False,
        model_sha256=audit.digest(path),
    )
    audit.save_json(audit.OUT / "plank_world_trial_results.json", report)
    return report


if __name__ == "__main__":
    print(run()["metrics"])
