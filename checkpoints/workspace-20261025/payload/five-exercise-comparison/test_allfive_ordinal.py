"""추가 AI 선택의 검증 역할 제한과 관측 열 계약을 검사한다."""

import unittest
from allfive_round1_ordinal import choose, estimator


class OrdinalSelectionTests(unittest.TestCase):
    """후보 선택이 검증 등급이나 숨은 특징을 사용하지 않게 한다."""

    def test_validation_cannot_choose_ai_policy(self):
        """검증 역할이 내부 모델 선택에 들어오면 거부한다."""
        with self.assertRaises(ValueError):
            choose([dict(split="validation")])

    def test_feature_indices_must_exist(self):
        """존재하지 않는 특징 열을 조용히 만들지 않는다."""
        with self.assertRaises(ValueError):
            estimator("regression", [11], 11)

    def test_unknown_model_kind_is_rejected(self):
        """명시하지 않은 예측 방식을 사용하지 않는다."""
        with self.assertRaises(ValueError):
            estimator("provider_grade", [0], 1)


if __name__ == "__main__":
    unittest.main()
