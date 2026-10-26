"""W 규칙에서 투영 각도를 실제 굽힘 정답으로 쓰지 않는지 검사한다."""

import unittest

import rule_score
import test_phases
import w_diagnostics


class WContractTests(unittest.TestCase):
    """투영의 비식별성과 등급 계약을 검사한다. 단위는 도·0~2등급이다."""

    def test_same_projection_different_true_angles(self) -> None:
        """동일한 화면 각도에서 실제 각도를 하나로 확정할 수 없어야 한다."""
        evidence = w_diagnostics.projection_counterexample()
        self.assertAlmostEqual(evidence["projected_angle_deg"], 150)
        self.assertAlmostEqual(evidence["lifted_3d_angle_deg"], 90)
        self.assertAlmostEqual(evidence["flat_3d_angle_deg"], 150)

    def test_complete_attempt_uses_morphology_not_90_target(self) -> None:
        """단계가 검출되어도 화면 각도 대리 등급을 공식 등급으로 쓰지 않는다."""
        sequence = test_phases.PhaseTest().sequence()
        result = rule_score.evaluate(
            "ytw",
            sequence,
            None,
            ["y_reach", "t_reach", "w_shape", "hinge_and_trunk_hold"],
            [30, 25, 25, 20],
        )
        self.assertTrue(
            result["evidence"]["phase_detection"]["complete_attempt_sequence"]
        )
        self.assertNotIn("target_deg", result["evidence"]["w_shape"])
        self.assertIsNone(result["score"])
        self.assertIn("morphology", result["evidence"]["w_shape"])


if __name__ == "__main__":
    unittest.main()
