"""이 구성요소의 입력·출력 및 실패 처리 계약을 검증합니다. 합성 입력이나 저장된 시험 자료의 검사 범위는 각 테스트 이름과 단언문에 명시합니다.

Meaningful checks for all newly trained model artifacts and inference contracts.
"""

import json
import unittest

import numpy as np
import torch
from suite_config import CONFIG_VERSION, SUITE, TASKS
from suite_features import make_features
from suite_inference import ExerciseClassifier, FiveExerciseClassifier


class SuiteTests(unittest.TestCase):
    def test_all_four_have_distinct_worker_splits(self):
        for exercise in TASKS:
            with self.subTest(exercise=exercise):
                out = SUITE / exercise / "outputs"
                s = json.loads((out / "splits.json").read_text())
                workers = {k: set(v["workers"]) for k, v in s.items()}
                self.assertFalse(
                    workers["train"] & workers["validation"]
                    or workers["train"] & workers["test"]
                    or workers["validation"] & workers["test"]
                )
                source = {
                    r["video_id"]: r
                    for r in json.loads(
                        (SUITE / exercise / "selected.json").read_text()
                    )
                }
                self.assertTrue(
                    all(
                        source[v]["source_split"] == "test"
                        for v in s["test"]["video_ids"]
                    )
                )
                self.assertTrue(
                    all(
                        source[v]["source_split"] == "train"
                        for v in s["train"]["video_ids"]
                    )
                )

    def test_preprocessing_and_serving_probability_parity(self):
        for exercise in TASKS:
            with self.subTest(exercise=exercise):
                out = SUITE / exercise / "outputs"
                s = json.loads((out / "splits.json").read_text())
                vid = s["test"]["video_ids"][0]
                model = ExerciseClassifier(exercise)
                with np.load(SUITE / "cache" / f"{vid}.npz") as cache:
                    x, raw, aspect = (
                        cache["X"],
                        cache["landmarks"],
                        float(cache["aspect"]),
                    )
                result = model.predict_landmarks(raw, aspect)
                self.assertEqual(result["status"], "ok")
                with torch.inference_mode():
                    expected = torch.sigmoid(
                        model.model(
                            torch.tensor(
                                ((x - model.mean) / model.std)[None],
                                dtype=torch.float32,
                            )
                        )
                    ).numpy()[0]
                for j, target in enumerate(model.targets):
                    if model.active[j]:
                        self.assertAlmostEqual(
                            result["scores"][target],
                            float(expected[j]),
                            places=5,
                        )
                    else:
                        self.assertIsNone(result["scores"][target])
                if not TASKS[exercise]["normal_supported"]:
                    self.assertNotIn("normal_candidate", result)

    def test_missing_pose_cannot_report_normal(self):
        for exercise in TASKS:
            model = ExerciseClassifier(exercise)
            result = model.predict_landmarks(np.zeros((32, 33, 4), np.float32))
            self.assertEqual(result["status"], "insufficient_pose_visibility")
            self.assertNotIn("normal_candidate", result)

    def test_unsupported_test_classes_have_null_f1(self):
        for exercise in TASKS:
            report = json.loads(
                (SUITE / exercise / "outputs/metrics.json").read_text()
            )
            for metrics in report["models"].values():
                for values in metrics["per_target"].values():
                    if not values["evaluable"]:
                        self.assertIsNone(values["f1"])
                if metrics["evaluated_targets"] < metrics["total_targets"]:
                    self.assertIsNone(metrics["macro_f1_all_targets"])

    def test_low_visibility_coordinates_are_masked(self):
        p = np.ones((8, 33, 4), np.float32)
        p[:, 23, 1] = 0.7
        p[:, 24, 1] = 0.7
        p[:, 11, 1] = 0.3
        p[:, 12, 1] = 0.3
        p[:, 27, 3] = 0.01
        x = make_features(p, 1.0)
        from suite_config import JOINTS

        start = JOINTS.index(27) * 4
        np.testing.assert_array_equal(x[:, start : start + 3], 0)
        self.assertTrue(np.isfinite(x).all())

    def test_exercise_is_required_and_validated(self):
        with self.assertRaises(ValueError):
            ExerciseClassifier("unknown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
