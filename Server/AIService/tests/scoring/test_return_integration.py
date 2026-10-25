"""시작·복귀 상태와 별도 등급 결합 계약을 검증합니다.

A ready-state classifier must not manufacture a normal or failure grade."""

import copy
import unittest

import numpy as np

from app.scoring.return_scoring import combine_return_score


class FrozenGradeModel:
    n_features_in_ = 2

    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)


class ReturnIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.score = dict(
            exercise="pushups",
            feature_version="reviewed-v1",
            ai_score=100.0,
            rule_score=100.0,
            items=[
                dict(
                    id="return_extension",
                    bank="side",
                    weight=100,
                    ai_grade=2.0,
                    rule_grade=2,
                    observed=True,
                    feature=[1.0, 2.0],
                    reasons=[],
                )
            ],
        )
        self.evidence = dict(
            schema="learned_initial_pose_return_evidence_v1",
            state=dict(return_events=1, status="observation_ended"),
            frames=[dict(predicted_ready_state=1)],
        )
        self.key = ("pushups", "return_extension", "side")
        self.head = dict(
            model=FrozenGradeModel(1.2),
            grade_support={1: 3, 2: 8},
            candidate_gate_passed=True,
            feature_dimensions=2,
            feature_version="reviewed-v1",
        )

    def run_case(self, **kw):
        return combine_return_score(
            self.score,
            self.evidence,
            {self.key: self.head},
            complete_attempt=kw.pop("complete_attempt", True),
            **kw,
        )

    def test_return_observed_does_not_override_learned_partial_grade(self):
        out = self.run_case()
        self.assertAlmostEqual(out["items"][0]["ai_grade"], 1.2)
        self.assertAlmostEqual(out["ai_score"], 60.0)
        self.assertEqual(
            out["return_grade_integration"]["unseen_training_grades"], [0]
        )
        self.assertEqual(self.score["ai_score"], 100.0)

    def test_normal_only_model_cannot_score_a_return(self):
        self.head["grade_support"] = {2: 10}
        self.assertIsNone(self.run_case()["items"][0]["ai_grade"])

    def test_failed_development_gate_holds_model(self):
        self.head["candidate_gate_passed"] = False
        self.assertIsNone(self.run_case()["ai_score"])

    def test_no_severity_head_does_not_turn_ready_into_two(self):
        out = combine_return_score(
            self.score, self.evidence, {}, complete_attempt=True
        )
        self.assertIsNone(out["items"][0]["ai_grade"])
        self.assertEqual(out["items"][0]["rule_grade"], 2)

    def test_user_stop_does_not_become_failure(self):
        out = self.run_case(user_stopped=True)
        self.assertIsNone(out["ai_score"])
        self.assertIsNone(out["rule_score"])
        self.assertIsNone(out["items"][0]["ai_grade"])

    def test_unreviewed_completion_is_not_failed_return(self):
        self.assertIsNone(self.run_case(complete_attempt=False)["ai_score"])

    def test_missing_state_is_not_failed_return(self):
        self.evidence["frames"].append(dict(predicted_ready_state=None))
        self.assertIsNone(self.run_case()["items"][0]["ai_grade"])

    def test_untrained_grade_zero_is_not_extrapolated(self):
        self.head["model"] = FrozenGradeModel(-0.5)
        self.assertEqual(self.run_case()["items"][0]["ai_grade"], 1.0)

    def test_normal_grade_requires_exactly_one_verified_return(self):
        self.head["model"] = FrozenGradeModel(2.0)
        for n in [0, 2]:
            self.evidence["state"]["return_events"] = n
            self.assertIsNone(self.run_case()["items"][0]["ai_grade"])
        self.evidence["state"]["return_events"] = 1
        self.assertEqual(self.run_case()["items"][0]["ai_grade"], 2.0)

    def test_earlier_return_does_not_validate_unfinished_final_attempt(self):
        self.head["model"] = FrozenGradeModel(2.0)
        self.evidence["state"]["status"] = "closed_attempt_return_not_observed"
        self.assertIsNone(self.run_case()["items"][0]["ai_grade"])

    def test_equal_dimensions_do_not_prove_equal_feature_version(self):
        self.head["feature_version"] = "unverified-other-v1"
        self.assertIsNone(self.run_case()["items"][0]["ai_grade"])

    def test_feature_dimension_mismatch_is_rejected(self):
        self.head["feature_dimensions"] = 3
        with self.assertRaises(ValueError):
            self.run_case()

    def test_knee_extension_is_not_a_standing_height_return_grade(self):
        self.score["exercise"] = "squats"
        out = self.run_case()
        self.assertIsNone(out["ai_score"])
        self.assertIsNone(out["return_grade_integration"]["grade"])
        self.assertEqual(
            out["return_grade_integration"]["criterion"],
            "standing_height_return",
        )

    def test_multiple_cycles_are_not_one_partial_return_training_unit(self):
        self.evidence["state"]["return_events"] = 2
        self.assertIsNone(self.run_case()["items"][0]["ai_grade"])


if __name__ == "__main__":
    unittest.main()
