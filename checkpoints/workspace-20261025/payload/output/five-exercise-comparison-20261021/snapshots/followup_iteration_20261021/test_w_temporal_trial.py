"""시간 W 집계의 짧은 구간·미관측·짧은 자세 이탈 처리를 검사한다."""

import unittest

import numpy as np

import w_temporal_trial


class WTemporalTests(unittest.TestCase):
    """합성 좌표로 결측을 점수로 바꾸지 않는지 확인한다."""

    def sequence(self) -> np.ndarray:
        """양팔 굽힘·손 위쪽과 몸통 길이 100픽셀인 5표본을 반환한다."""
        seq = np.zeros((5, 33, 2), dtype=float)
        seq[:, 11], seq[:, 12] = [20, 0], [-20, 0]
        seq[:, 23], seq[:, 24] = [20, 100], [-20, 100]
        seq[:, 13], seq[:, 14] = [70, 0], [-70, 0]
        seq[:, 15], seq[:, 16] = [70, -50], [-70, -50]
        return seq

    def test_brief_deviation_preserves_dominant_shape(self) -> None:
        """5표본 중 한 장 팔을 편 이탈이 유지 모양 전체를 뒤집지 않는다."""
        seq = self.sequence()
        seq[2, 15], seq[2, 16] = [120, 0], [-120, 0]
        self.assertEqual(w_temporal_trial.evaluate(seq, 0, 4)["grade"], 2)

    def test_occlusion_is_not_partial_or_good(self) -> None:
        """40% 손목 결측은 부분 점수나 만점으로 채우지 않는다."""
        seq = self.sequence()
        seq[:2, 15] = np.nan
        self.assertIsNone(w_temporal_trial.evaluate(seq, 0, 4)["grade"])

    def test_single_frame_is_not_dwell(self) -> None:
        """내부 계산의 반복 프레임을 실제 유지 관측으로 세지 않는다."""
        self.assertIsNone(
            w_temporal_trial.evaluate(self.sequence(), 0, 0)["grade"]
        )
