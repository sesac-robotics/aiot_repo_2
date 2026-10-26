"""누락·구도·단위·파일 지문과 실제 번들의 점수 회귀를 검사한다."""

import json
import pathlib
import tempfile
import unittest
from typing import Any, cast

import numpy as np

from app.scoring import engine, features, held_geometry, rules
from app.scoring.__main__ import demo_sequence
from app.scoring.sequence import PoseSequence, verify_pair


class SequenceTest(unittest.TestCase):
    """관절 좌표와 시각의 실패 계약을 검사한다."""

    def test_duplicate_frames_rejected(self) -> None:
        """같은 시각 두 번 입력을 시계열로 사용하지 않는다."""
        with self.assertRaises(ValueError):
            PoseSequence(np.zeros((2, 33, 2)), np.array([0, 0]), 30, 640, 480)

    def test_three_dimensions_rejected(self) -> None:
        """3D world 좌표를 2D 픽셀 특징으로 잘못 사용하지 않는다."""
        with self.assertRaises(ValueError):
            PoseSequence(
                np.zeros((3, 33, 3)), np.array([0, 3, 6]), 30, 640, 480
            )

    def test_missing_coordinate_masks_entire_joint(self) -> None:
        """한 축만 누락된 점도 추측해서 채우지 않는다."""
        seq = demo_sequence()
        xy = seq.image_xy.copy()
        xy[0, 11, 0] = np.nan
        result = PoseSequence(xy, seq.frame_numbers, 30, 640, 480)
        self.assertTrue(np.isnan(result.image_xy[0, 11]).all())
        self.assertTrue(np.isfinite(seq.image_xy[0, 11]).all())

    def test_sampling_rate_rejected(self) -> None:
        """모델과 다른 1Hz 캐시를 조용히 10Hz로 사용하지 않는다."""
        with self.assertRaises(ValueError):
            PoseSequence(
                np.zeros((3, 33, 2)), np.array([0, 30, 60]), 30, 640, 480
            )

    def test_pair_shift_checked(self) -> None:
        """같은 길이여도 원본 프레임 시차가 다르면 거부한다."""
        seq = demo_sequence()
        front = PoseSequence(seq.image_xy, seq.frame_numbers + 5, 30, 640, 480)
        verify_pair(seq, front, 5)
        with self.assertRaises(ValueError):
            verify_pair(seq, front, 0)


class EngineTest(unittest.TestCase):
    """선택 모델 5개를 실제 로드해 중요한 미판정·배점 동작을 검사한다."""

    def setUp(self) -> None:
        """네트워크·카메라 없이 로컬 검증 번들 계산기를 생성한다."""
        self.scorer = engine.ScoringEngine()

    def test_five_missing_sequences_never_become_zero(self) -> None:
        """관절 누락은 오류 0점이나 기본 만점이 되지 않는다."""
        seq = PoseSequence(
            np.full((31, 33, 2), np.nan), np.arange(31) * 3, 30, 640, 480
        )
        for exercise in features.DEFINITIONS:
            with self.subTest(exercise=exercise):
                result = self.scorer.score(
                    exercise,
                    seq,
                    view="front" if exercise in ("shoulder", "ytw") else "side",
                    complete_attempt=True,
                )
                self.assertIsNone(result["ai_score"])
                self.assertIsNone(result["rule_score"])
                self.assertTrue(
                    all(
                        x["ai_grade"] is None and x["rule_grade"] is None
                        for x in result["items"]
                    )
                )

    def test_shoulders_do_not_require_face_or_legs(self) -> None:
        """양팔·몸통만 필요한 구도에서는 얼굴·하체 누락을 전신 거부로 확대하지 않는다."""
        result = self.scorer.score(
            "shoulder", demo_sequence(), view="front", complete_attempt=True
        )
        self.assertTrue(any(x["ai_grade"] is not None for x in result["items"]))
        self.assertFalse(
            any("face" in str(x["reasons"]) for x in result["items"])
        )

    def test_uncompleted_attempt_has_items_but_no_total(self) -> None:
        """보이는 항목과 완결 반복 총점의 승인을 분리한다."""
        result = self.scorer.score("shoulder", demo_sequence(), view="front")
        self.assertTrue(any(x["ai_grade"] is not None for x in result["items"]))
        self.assertIsNone(result["ai_score"])
        self.assertIsNone(result["rule_score"])

    def test_unreviewed_items_remove_total(self) -> None:
        """부족한 항목을 나머지 항목 점수로 100점 환산하지 않는다."""
        result = self.scorer.score(
            "shoulder",
            demo_sequence(),
            view="front",
            complete_attempt=True,
            permitted_items=[True, True, True, False],
        )
        self.assertIsNone(result["items"][3]["ai_grade"])
        self.assertIsNone(result["ai_score"])

    def test_oblique_trunk_is_not_assessed(self) -> None:
        """사선 몸통의 실제 좌우 기울기를 정면 값으로 확정하지 않는다."""
        result = self.scorer.score(
            "shoulder",
            demo_sequence(),
            view="front_oblique",
            complete_attempt=True,
        )
        self.assertIsNone(result["items"][3]["ai_grade"])
        self.assertIsNone(result["rule_score"])

    def test_strings_cannot_authorize_review(self) -> None:
        """JSON 문자열 false가 참으로 해석되어 총점을 승인하지 않는다."""
        with self.assertRaises(ValueError):
            self.scorer.score(
                "shoulder",
                demo_sequence(),
                view="front",
                complete_attempt=cast(Any, "false"),
            )

    def test_invalid_phase_intervals_rejected(self) -> None:
        """음수·역순·소수 프레임 구간을 단계 승인에 사용하지 않는다."""
        for interval in ([-1, 10], [10, 2], [1.5, 10]):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                self.scorer.score(
                    "ytw",
                    demo_sequence(),
                    view="front",
                    phase_windows={"w_shape": interval},
                )

    def test_paired_view_needs_same_attempt_review(self) -> None:
        """정렬 번호만 같다고 서로 다른 시범을 합치지 않는다."""
        with self.assertRaises(ValueError):
            self.scorer.score(
                "squats", demo_sequence(), view="side", front=demo_sequence()
            )

    def test_ytw_requires_projection_review(self) -> None:
        """정면만으로 전경·Y가 보인다고 가정하여 총점을 만들지 않는다."""
        result = self.scorer.score(
            "ytw", demo_sequence(), view="front", complete_attempt=True
        )
        self.assertIsNone(result["items"][0]["ai_grade"])
        self.assertIsNone(result["items"][3]["ai_grade"])
        self.assertIsNone(result["ai_score"])

    def test_repeated_inference_is_deterministic(self) -> None:
        """같은 입력을 재평가해도 시간 값 외 출력은 변하지 않는다."""
        first = self.scorer.score(
            "shoulder", demo_sequence(), view="front", complete_attempt=True
        )
        second = self.scorer.score(
            "shoulder", demo_sequence(), view="front", complete_attempt=True
        )
        first.pop("timing_ms")
        second.pop("timing_ms")
        self.assertEqual(first, second)

    def test_bundle_tampering_rejected_before_deserialization(self) -> None:
        """지문이 다른 모델 파일은 읽기 전에 거부한다."""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            manifest = dict(
                schema="posture-score-bundle-v1",
                exercises={
                    "shoulder": dict(file="shoulder.joblib", sha256="0" * 64)
                },
            )
            (path / "manifest.json").write_text(json.dumps(manifest))
            (path / "shoulder.joblib").write_bytes(b"not a model")
            with self.assertRaises(ValueError):
                engine.ScoringEngine(path).model("shoulder")

    def test_pushup_weights_match_review_rubric(self) -> None:
        """부위별 배점을 원래 채점표와 동일하게 유지한다."""
        self.assertEqual(features.DEFINITIONS["pushups"][1], [40, 30, 20, 10])

    def test_observed_zero_is_not_unknown(self) -> None:
        """관측된 명시 오류 0등급과 미판정을 구분한다."""
        note = dict(
            eligible=True,
            measurement=50,
            higher_is_better=False,
            item="trunk_balance",
        )
        self.assertEqual(rules.rule_grade(note, (8, 15)), 0)
        note["eligible"] = False
        self.assertIsNone(rules.rule_grade(note, (8, 15)))


if __name__ == "__main__":
    unittest.main()


class TrainingSupportTest(unittest.TestCase):
    """점수와 별도로 부족한 등급의 학습 근거를 표시하는지 검사한다."""

    def test_missing_grades_are_visible_without_fabricated_examples(
        self,
    ) -> None:
        """정상 등급만 배운 하단 복귀의 오류 사례 수를 0 그대로 공개한다."""
        result = engine.ScoringEngine().score(
            "shoulder", demo_sequence(), view="front", complete_attempt=True
        )
        item = next(
            row for row in result["items"] if row["id"] == "lower_return"
        )
        self.assertEqual(item["training_grade_counts"]["0"], 0)
        self.assertEqual(item["training_grade_counts"]["1"], 0)
        self.assertGreater(item["training_grade_counts"]["2"], 0)
        self.assertEqual(item["unseen_training_grades"], [0, 1])
        self.assertIsNotNone(item["ai_grade"])
        self.assertTrue(item["single_grade_training"])
