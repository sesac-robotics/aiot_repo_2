"""준비 상태 근거와 별도로 학습된 복귀 등급을 안전하게 결합합니다.

Combine learned state evidence and a separately trained return severity head.

An observed return never supplies a constant grade. A partial or missing return
needs a trained severity head and a reviewed complete attempt, not a timeout.
"""

import copy

import numpy as np

RETURN_ITEMS = {
    "pushups": "return_extension",
    "shoulder": "lower_return",
    "squats": "standing_height_return",
}


def combine_return_score(
    score, evidence, heads, *, complete_attempt, user_stopped=False
):
    """Use trusted local candidate objects only, with exact feature-bank matching.

    This experimental adapter never fits or loads an arbitrary pickle. Grade
    support and adoption gates come from frozen training metadata. State models
    and severity models remain distinct, including their reported limitations.
    """
    result = copy.deepcopy(score)
    result["initial_return_evidence"] = evidence
    result["production_replacement"] = False
    exercise = score["exercise"]
    item_id = RETURN_ITEMS.get(exercise)
    if item_id is None:
        result["return_grade_integration"] = dict(
            status="hold_or_phase_order_exercise_not_repetition_return",
            grade=None,
            state_is_not_quality_grade=True,
        )
        return result
    item = next((i for i in result["items"] if i["id"] == item_id), None)
    if item is None:
        # Legacy squat knee extension and standing height have different targets.
        result["ai_score"] = result["ai_minus_rule"] = None
        result["return_grade_integration"] = dict(
            status="unassessed",
            grade=None,
            holds=["return_target_not_equivalent_to_legacy_score_item"],
            criterion=item_id,
            state_is_not_quality_grade=True,
        )
        return result
    held = []
    state = evidence.get("state", {})
    if evidence.get("schema") != "learned_initial_pose_return_evidence_v1":
        held.append("learned_state_evidence_not_available")
    if not complete_attempt:
        held.append("complete_attempt_not_reviewed")
    if user_stopped:
        held.append("user_stop_is_not_return_failure")
    if state.get("return_events", 0) > 1:
        held.append("multiple_return_cycles_need_separate_attempt_windows")
    if evidence.get("status", "").startswith("unassessed"):
        held.append(evidence["status"])
    frames = evidence.get("frames", [])
    if not frames or any(
        f.get("predicted_ready_state") is None for f in frames
    ):
        held.append("return_attempt_has_unobserved_state")
    key = (exercise, item_id, item["bank"])
    head = heads.get(key)
    if not item.get("observed") or item.get("feature") is None:
        held.append("return_geometry_not_observed")
    if head is None:
        held.append("return_severity_head_not_trained_for_feature_bank")
    else:
        counts = head.get("grade_support", head.get("grade_counts", {}))
        counts = {int(g): int(n) for g, n in counts.items() if n > 0}
        if counts.get(2, 0) == 0 or not any(g < 2 for g in counts):
            held.append("return_severity_normal_and_error_training_missing")
        if head.get("candidate_gate_passed") is not True:
            held.append("return_severity_development_gate_not_passed")
        if head.get("feature_version") != score.get(
            "feature_version"
        ) or not score.get("feature_version"):
            held.append("return_severity_feature_version_not_verified")
    grade = None
    if not held:
        feature = np.asarray(item["feature"], float)
        model = head["model"]
        if (
            feature.ndim != 1
            or not np.isfinite(feature).all()
            or len(feature) != head["feature_dimensions"]
            or len(feature) != model.n_features_in_
        ):
            raise ValueError("Return severity feature schema differs")
        raw = float(model.predict([feature])[0])
        if not np.isfinite(raw):
            raise ValueError("Nonfinite return severity prediction")
        # No extrapolated grade0 when only grades1/2 were reviewed and trained.
        grade = float(np.clip(raw, min(counts), max(counts)))
        if grade >= 1.5 and (
            state.get("return_events") != 1
            or state.get("status")
            in (
                "closed_attempt_return_not_observed",
                "return_unverified",
                "stopped_by_user",
                "unobserved",
            )
        ):
            held.append("normal_grade_without_verified_temporal_return")
            grade = None
    item["ai_grade"] = grade
    item["ai_minus_rule"] = (
        None
        if grade is None or item.get("rule_grade") is None
        else grade - item["rule_grade"]
    )
    item["reasons"] = list(dict.fromkeys(item.get("reasons", []) + held))
    item["initial_return_status"] = state.get("status", evidence.get("status"))
    item["severity_is_separately_learned"] = True
    result["return_grade_integration"] = dict(
        status="unassessed" if held else "trained_severity_with_state_evidence",
        grade=grade,
        holds=held,
        state_is_not_quality_grade=True,
        unseen_training_grades=(
            [g for g in range(3) if g not in counts]
            if head is not None
            else [0, 1, 2]
        ),
    )
    values = [i["ai_grade"] for i in result["items"]]
    result["ai_score"] = (
        float(sum(i["weight"] * i["ai_grade"] / 2 for i in result["items"]))
        if complete_attempt and not user_stopped and None not in values
        else None
    )
    if user_stopped:
        result["rule_score"] = None
    result["ai_minus_rule"] = (
        None
        if result["ai_score"] is None or result.get("rule_score") is None
        else result["ai_score"] - result["rule_score"]
    )
    return result
