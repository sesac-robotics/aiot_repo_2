"""정지 항목 비교의 관측·정규화·참여자 분리 경계를 검사한다."""

import unittest

import numpy as np

from plank_static_benchmark import check_groups, pose_features


class PlankStaticTest(unittest.TestCase):
    """미관측과 동일 원본의 교차 분할을 허용하지 않는지 검사한다."""

    def test_translation_scale_and_reflection_preserve_features(self) -> None:
        """같은 자세의 이동·확대·좌우 반전은 특징을 바꾸지 않는다."""
        xy = np.full((33, 2), np.nan)
        xy[[11, 23, 27]] = [[40, 50], [100, 40], [210, 80]]
        confidence = np.ones((33, 2))
        original = pose_features(xy, confidence)
        changed = xy * [-2, 2] + [600, 110]
        derived = pose_features(changed, confidence)
        np.testing.assert_allclose(original["features"], derived["features"])
        self.assertAlmostEqual(original["angle_deg"], derived["angle_deg"])

    def test_missing_or_untrusted_ankle_holds_evaluation(self) -> None:
        """어깨·골반만 보여도 발목이 없거나 낮은 신뢰도면 채점하지 않는다."""
        xy = np.full((33, 2), np.nan)
        xy[[11, 23]] = [[10, 20], [50, 20]]
        confidence = np.ones((33, 2))
        self.assertIsNone(pose_features(xy, confidence))
        xy[27] = [100, 20]
        confidence[27, 0] = 0.49
        self.assertIsNone(pose_features(xy, confidence))

    def test_same_image_cannot_cross_person_groups(self) -> None:
        """같은 원본을 다른 참여자 ID로 분리하려 하면 실패한다."""
        records = [
            dict(id="a", person_token="P1", image_sha256="same"),
            dict(id="b", person_token="P2", image_sha256="same"),
        ]
        with self.assertRaises(ValueError):
            check_groups(records)


if __name__ == "__main__":
    unittest.main()
