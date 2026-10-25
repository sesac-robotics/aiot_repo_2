"""뒤 반복의 긴 유지가 첫 시행을 덮어쓰지 않는지 검사한다."""

import unittest
import numpy as np

import test_phases
import ytw_phases
import ytw_first_cycle_trial


class FirstCycleTests(unittest.TestCase):
    """가변 유지 시간과 누락 관절에서 시행 선택 계약을 검증한다."""

    def test_long_second_cycle_does_not_replace_first(self) -> None:
        """같은 모양의 두 번째 시행이 길어도 첫 완전 시행을 선택한다."""
        first = test_phases.PhaseTest().sequence()
        sequence = np.concatenate([first, np.repeat(first, 2, axis=0)])
        baseline = ytw_phases.detect(sequence)
        result = ytw_first_cycle_trial.detect(sequence)
        self.assertGreater(
            baseline["phases"]["W_attempt"]["start_index"], len(first)
        )
        self.assertLess(result["phases"]["W_attempt"]["end_index"], len(first))

    def test_occlusion_does_not_invent_cycle(self) -> None:
        """모든 팔 관측이 사라지면 첫 시행을 임의로 만들지 않는다."""
        sequence = test_phases.PhaseTest().sequence()
        sequence[:, [15, 16]] = np.nan
        self.assertFalse(
            ytw_first_cycle_trial.detect(sequence)["complete_attempt_sequence"]
        )
