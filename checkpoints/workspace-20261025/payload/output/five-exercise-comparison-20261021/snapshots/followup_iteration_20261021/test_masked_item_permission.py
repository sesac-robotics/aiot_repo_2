"""부분 항목 예측에서 구도 허용과 미관측 총점 보류 계약을 검사한다."""

import unittest
from unittest.mock import Mock

import numpy as np

from plank_masked_multitask_trial import predict


class MaskedPermissionTests(unittest.TestCase):
    """관절이 보여도 구도 허용이 없으면 점수를 만들지 않는지 검사한다."""

    def test_no_view_permission_does_not_score(self):
        """기본 호출은 촬영 구도를 추정하여 임의로 모든 항목을 허용하지 않는다."""
        sequence = np.ones((30, 33, 2))
        result = predict([Mock()] * 4, sequence, 3)
        self.assertEqual(result["ai_levels"], [None] * 4)
        self.assertIsNone(result["ai_total"])
        self.assertIsNone(result["rule_total"])

    def test_visible_elbow_does_not_fill_other_items(self):
        """상지 항목만 평가한 등급을 전신 만점으로 환산하지 않는다."""
        sequence = np.full((30, 33, 2), np.nan)
        for joint, point in [
            (11, (200, 100)),
            (13, (200, 150)),
            (23, (400, 120)),
        ]:
            sequence[:, joint] = point
        model = Mock()
        model.predict.return_value = np.array([2.0])
        result = predict(
            [model] * 4,
            sequence,
            3,
            permitted_items=[False, True, False, False],
        )
        self.assertEqual(result["ai_levels"], [None, 2.0, None, None])
        self.assertIsNone(result["ai_total"])
        self.assertIsNone(result["rule_total"])
