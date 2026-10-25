"""기존 무릎 각도 혼동과 카메라 크기·잘림·불완전 반복 회귀를 검사한다."""

import unittest

import numpy as np

from held_geometry import (
    plank_clearance,
    plank_support_grade,
    squat_foot_change,
)


class HeldGeometryTests(unittest.TestCase):
    """동일 무릎 각도의 다른 지지 상태와 상·하단 발 변화 계약을 검사한다."""

    def support(self, rest=False):
        """같은 편 무릎을 가진 지지 또는 바닥 상태의 합성 픽셀을 반환한다."""
        seq = np.full((12, 33, 2), np.nan)
        coords = {
            11: (30, 40),
            13: (30, 100),
            23: (80, 60),
            25: (140, 80),
            27: (200, 100),
            31: (210, 100),
        }
        if rest:
            coords.update({23: (80, 98), 25: (140, 99)})
        for i, xy in coords.items():
            seq[:, i] = xy
        return seq

    def foot(self, lift=False):
        """상단 편 무릎과 하단 굽힌 무릎의 고정 또는 들린 발을 반환한다."""
        seq = np.full((20, 33, 2), np.nan)
        seq[:, 23] = [0, 0]
        seq[:, 25] = [0, 20]
        seq[:, 27] = [0, 40]
        seq[10:, 27] = [20, 20]
        seq[:, 29] = [0, 45]
        seq[:, 31] = [20, 45]
        if lift:
            seq[10:, 29, 1] = 35
        return seq

    def test_straight_knee_rest_is_not_normal_support(self):
        """편 무릎만 같은 바닥 상태에 새 규칙이 만점을 주지 않는다."""
        normal, rest = plank_clearance(self.support()), plank_clearance(
            self.support(True)
        )
        self.assertAlmostEqual(normal["measurement"][0], rest["measurement"][0])
        self.assertEqual(plank_support_grade(normal["measurement"]), 2)
        self.assertEqual(plank_support_grade(rest["measurement"]), 0)

    def test_support_translation_reflection_and_scale(self):
        """같은 지지는 영상 이동·배율·좌우 반전으로 등급이 바뀌지 않는다."""
        seq = self.support()
        ref = plank_clearance(seq)
        mirror = seq.copy()
        mirror[:, :, 0] = 300 - mirror[:, :, 0]
        for transformed in (seq * 3 + 42, mirror):
            self.assertTrue(
                np.allclose(
                    ref["feature"], plank_clearance(transformed)["feature"]
                )
            )

    def test_missing_contact_points_are_not_filled(self):
        """발끝 누락이 20%를 넘으면 바닥 접촉 대리값을 만들지 않는다."""
        seq = self.support()
        seq[:3, 31] = np.nan
        self.assertFalse(plank_clearance(seq)["eligible"])

    def test_standing_to_bottom_heel_change(self):
        """발의 고정 위치와 하단 뒤꿈치 상승이 다른 특징으로 나타난다."""
        still = squat_foot_change(self.foot())
        raised = squat_foot_change(self.foot(True))
        self.assertEqual(still["feature"][-1], 0)
        self.assertGreater(raised["feature"][-1], 0.25)
        self.assertGreater(raised["feature"][2], 20)

    def test_foot_change_is_not_torso_foreshortening(self):
        """필요하지 않은 어깨 위치·영상 배율이 발 상승 비율을 바꾸지 않는다."""
        seq = self.foot(True)
        expected = squat_foot_change(seq)["feature"]
        seq[:, 11] = [400, -200]
        self.assertTrue(
            np.allclose(expected, squat_foot_change(seq * 2 + 31)["feature"])
        )

    def test_no_observed_squat_bottom_abstains(self):
        """굽힌 하단 없이 서 있기만 한 구간을 발 정상 예시로 만들지 않는다."""
        seq = self.foot()
        seq[:, 27] = [0, 40]
        self.assertFalse(squat_foot_change(seq)["eligible"])

    def test_partial_return_still_observes_foot_change(self):
        """다 펴지 않은 복귀도 움직임·발 관측이 충분하면 발을 측정한다."""
        seq = self.foot(True)
        seq[:10, 27] = [14, 34]
        self.assertFalse(squat_foot_change(seq)["eligible"])
        result = squat_foot_change(seq, allow_partial_top=True)
        self.assertTrue(result["eligible"])
        self.assertGreater(result["feature"][-1], 0.25)

    def test_static_crouch_is_not_partial_repetition(self):
        """가만히 굽혀 있는 자세에 부분 복귀 경로가 정상 발 변화를 만들지 않는다."""
        seq = self.foot()
        seq[:, 27] = [20, 20]
        self.assertFalse(
            squat_foot_change(seq, allow_partial_top=True)["eligible"]
        )


if __name__ == "__main__":
    unittest.main()
