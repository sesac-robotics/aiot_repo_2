"""부분 구간 실험에서 미관측 Y·휴식·가림을 채우지 않는지 검사한다."""

import unittest
import numpy as np
import test_phases
import ytw_phases
import ytw_partial_trial


class PartialPhaseTests(unittest.TestCase):
    """실제 점수 라벨과 무관한 시계열로 부분 구간의 실패 조건을 검사한다."""

    def partial_sequence(self) -> np.ndarray:
        """Y 없이 T와 W 시도만 있는 N×33×2 픽셀 배열을 반환한다."""
        seq = test_phases.PhaseTest().sequence()
        seq[6:14, 15] = [-20, 130]
        seq[6:14, 16] = [20, 130]
        return seq

    def test_partial_w_does_not_invent_y(self) -> None:
        """T·W 관측만으로 Y 또는 완전한 세 단계를 만들지 않아야 한다."""
        result = ytw_partial_trial.detect(self.partial_sequence())
        self.assertTrue(result["partial_fallback_used"])
        self.assertIn("W_attempt", result["phases"])
        self.assertNotIn("Y_attempt", result["phases"])
        self.assertFalse(result["complete_attempt_sequence"])
        self.assertFalse(result["canonical_Y_depth_confirmed"])

    def test_rest_after_t_is_not_w(self) -> None:
        """T 뒤 양팔을 아래로 내린 휴식을 W로 채우지 않아야 한다."""
        seq = self.partial_sequence()
        seq[27:, 15] = [-20, 130]
        seq[27:, 16] = [20, 130]
        seq[27:, 13] = [-20, 65]
        seq[27:, 14] = [20, 65]
        self.assertNotIn("W_attempt", ytw_partial_trial.detect(seq)["phases"])

    def test_missing_w_arm_is_not_interpolated(self) -> None:
        """W 위치의 한 팔 누락을 보간해 단계로 만들지 않아야 한다."""
        seq = self.partial_sequence()
        seq[29:38, 15] = np.nan
        self.assertNotIn("W_attempt", ytw_partial_trial.detect(seq)["phases"])

    def test_complete_base_windows_preserved(self) -> None:
        """기존에 완전한 세 구간은 추가 실험이 변경하지 않아야 한다."""
        seq = test_phases.PhaseTest().sequence()
        self.assertEqual(
            ytw_phases.detect(seq)["phases"],
            ytw_partial_trial.detect(seq)["phases"],
        )

    def test_partial_window_rotation_invariance(self) -> None:
        """전체 화면을 90도 회전해도 부분 구간 위치가 같아야 한다."""
        seq = self.partial_sequence()
        self.assertEqual(
            ytw_partial_trial.detect(seq)["phases"],
            ytw_partial_trial.detect(seq @ np.array([[0, -1], [1, 0]]))[
                "phases"
            ],
        )


if __name__ == "__main__":
    unittest.main()
