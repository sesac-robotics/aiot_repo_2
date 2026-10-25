"""복구 후보의 분할 누출·미학습 구도·결측 평가 계약을 검증한다."""

import unittest

from recovery_train import component_groups, predict, summarize
from recovery_balance import weights_for
from recovery_variant import standing_allowed


class RecoveryContractTests(unittest.TestCase):
    """연결 원본·부분 점수에서 잘못된 성능 집계를 막는 회귀 검사다."""

    def test_seated_source_cannot_reenter_through_another_repetition_id(self):
        row = dict(
            exercise="shoulder",
            video_sha256="seated_source",
            id="unseen_repetition_24",
        )
        self.assertFalse(standing_allowed(row, {"seated_source"}))
        row["video_sha256"] = "standing_source"
        self.assertTrue(standing_allowed(row, {"seated_source"}))
        row["observed_variant"] = "seated_press"
        self.assertFalse(standing_allowed(row, set()))

    def test_incomplete_attempt_keeps_items_but_holds_both_totals(self):
        class ConstantModel:
            def predict(self, features):
                return [2.0] * len(features)

        row = dict(
            exercise="ytw",
            criterion_ids=["a", "b", "c", "d"],
            notes=[
                dict(bank="front", eligible=True, feature=[1], rule_grade=2)
                for _ in range(4)
            ],
            weights=[30, 25, 25, 20],
            whole_attempt_observed=False,
        )
        heads = {
            ("ytw", item, "front"): dict(model=ConstantModel(), boundary=None)
            for item in row["criterion_ids"]
        }
        result = predict(heads, row)
        self.assertEqual(result["ai_levels"], [2.0] * 4)
        self.assertIsNone(result["ai_score"])
        self.assertIsNone(result["rule_score"])

    def test_repeated_recording_connects_different_worker_metadata(self):
        rows = [
            dict(
                person="worker1",
                sources={"clip1"},
                split="train",
                recording_groups=["room1"],
            ),
            dict(
                person="worker2",
                sources={"clip2"},
                split="train",
                recording_groups=["room1"],
            ),
        ]
        self.assertEqual(len(set(component_groups(rows))), 1)
        rows[1]["split"] = "validation"
        with self.assertRaises(ValueError):
            component_groups(rows)

    def test_rare_grade_weight_is_capped_without_repeat_inflation(self):
        rows = [
            dict(person="rare", levels=[1]),
            dict(person="common", levels=[2]),
        ]
        original = weights_for(rows, 0, 3)
        repeated = weights_for(rows + [dict(person="rare", levels=[1])], 0, 3)
        self.assertAlmostEqual(original[0], repeated[0] + repeated[2])
        self.assertLessEqual(original[0], 1 / 3)

    def test_transitive_source_and_person_components(self):
        rows = [
            dict(person="a", sources={"one"}, split="train"),
            dict(person="b", sources={"one", "two"}, split="train"),
            dict(person="c", sources={"two"}, split="train"),
            dict(person="d", sources={"three"}, split="train"),
        ]
        groups = component_groups(rows)
        self.assertEqual(len(set(groups[:3])), 1)
        self.assertNotEqual(groups[0], groups[3])

    def test_cross_role_connected_sources_rejected(self):
        with self.assertRaises(ValueError):
            component_groups(
                [
                    dict(person="a", sources={"shared"}, split="train"),
                    dict(person="b", sources={"shared"}, split="validation"),
                ]
            )

    def test_missing_view_bank_never_fills_total(self):
        row = dict(
            exercise="pushups",
            criterion_ids=["a", "b", "c", "d"],
            notes=[
                dict(bank="front", eligible=True, feature=[1]) for _ in range(4)
            ],
            weights=[40, 30, 20, 10],
        )
        result = predict({}, row)
        self.assertEqual(result["ai_levels"], [None] * 4)
        self.assertIsNone(result["ai_score"])
        self.assertIsNone(result["rule_score"])

    def test_total_mae_uses_identical_paired_rows(self):
        rows = [
            dict(
                target_score=0,
                ai_score=10,
                rule_score=20,
                criterion_ids=[],
                levels=[],
                ai_levels=[],
                rule_levels=[],
            ),
            dict(
                target_score=100,
                ai_score=0,
                rule_score=None,
                criterion_ids=[],
                levels=[],
                ai_levels=[],
                rule_levels=[],
            ),
        ]
        report = summarize(rows)
        self.assertEqual(report["paired_total_records"], 1)
        self.assertEqual(
            report["paired_total_MAE_points"], dict(ai_score=10, rule_score=20)
        )
        self.assertEqual(report["ai_total_available"], 2)

    def test_missing_reference_not_a_zero_grade(self):
        row = dict(
            target_score=None,
            ai_score=0,
            rule_score=0,
            criterion_ids=["a"],
            levels=[None],
            ai_levels=[0],
            rule_levels=[0],
        )
        report = summarize([row])
        self.assertEqual(report["paired_total_records"], 0)
        self.assertEqual(report["items"], {})


if __name__ == "__main__":
    unittest.main()
