"""단계 누락·단발 잡음·화면 회전에서 시도 구간 인식의 경계를 검사한다."""

import unittest
import numpy as np

import ytw_phases


class PhaseTest(unittest.TestCase):
    """정답 점수와 무관한 시계열 단계 인식 실패 조건을 검사한다."""

    def sequence(self) -> np.ndarray:
        """몸통 100픽셀인 세 유지 구간의 합성 관절 배열을 반환한다."""
        seq = np.zeros((45, 33, 2))
        seq[:, 11] = [-20, 0]
        seq[:, 12] = [20, 0]
        seq[:, 23] = [-20, 100]
        seq[:, 24] = [20, 100]
        seq[:, 15] = [-20, 130]
        seq[:, 16] = [20, 130]
        for a, b, width, height in [
            (6, 14, 60, -30),
            (16, 27, 150, 0),
            (29, 38, 65, -20),
        ]:
            seq[a:b, 15] = [-width, height]
            seq[a:b, 16] = [width, height]
        return seq

    def test_dwell_order_and_rotation(self) -> None:
        """전체 화면 회전 후에도 같은 세 구간을 식별해야 한다."""
        seq = self.sequence()
        a = ytw_phases.detect(seq)
        b = ytw_phases.detect(seq @ np.array([[0, -1], [1, 0]]))
        self.assertTrue(a["complete_attempt_sequence"])
        self.assertEqual(a["phases"], b["phases"])
        self.assertFalse(a["canonical_Y_depth_confirmed"])

    def test_t_width_drift_does_not_become_w(self) -> None:
        """T 유지 중 80% 폭으로 줄어든 흔들림은 W로 잘라 내지 않는다."""
        seq = self.sequence()
        seq[16:29, 15] = [-150, 0]
        seq[16:29, 16] = [150, 0]
        seq[21:26, 15] = [-120, 0]
        seq[21:26, 16] = [120, 0]
        phases = ytw_phases.detect(seq)["phases"]
        self.assertGreaterEqual(phases["W_attempt"]["start_index"], 29)
        self.assertLessEqual(phases["T_attempt"]["start_index"], 20)
        self.assertGreaterEqual(phases["T_attempt"]["end_index"], 26)

    def test_missing_first_attempt_is_not_filled(self) -> None:
        """T·W만 있는 구간에 Y 시도를 만들어 넣지 않아야 한다."""
        seq = self.sequence()
        seq[6:14, 15] = [-20, 130]
        seq[6:14, 16] = [20, 130]
        self.assertFalse(ytw_phases.detect(seq)["complete_attempt_sequence"])

    def test_single_frame_peak_is_not_a_dwell(self) -> None:
        """단일 프레임 팔 벌림을 유지 단계로 채택하지 않아야 한다."""
        seq = self.sequence()
        seq[16:27, 15] = [-20, 130]
        seq[16:27, 16] = [20, 130]
        seq[20, 15] = [-150, 0]
        seq[20, 16] = [150, 0]
        self.assertFalse(ytw_phases.detect(seq)["complete_attempt_sequence"])

    def test_missing_arm_joints_are_not_interpolated(self) -> None:
        """필수 팔 관절 누락을 단계 시퀀스로 추측하지 않아야 한다."""
        seq = self.sequence()
        seq[:, 15] = np.nan
        self.assertFalse(ytw_phases.detect(seq)["complete_attempt_sequence"])


if __name__ == "__main__":
    unittest.main()
