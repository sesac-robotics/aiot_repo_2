"""캐시가 남아 있어도 확인된 앉은 프레스가 서서 하는 학습에 섞이지 않게 검사한다."""

import unittest
from unittest.mock import patch

import three_pipeline


class VariantContractTest(unittest.TestCase):
    """제공자 정상·오류 문자열과 독립인 직접 확인 운동 변형을 검사한다."""

    def test_seated_partial_labels_are_not_standing_fit_inputs(self) -> None:
        """관측한 팔 등급이 있어도 앉은 변형을 목표 운동으로 합치지 않는다."""
        row = dict(
            id="subject_011_shoulder_bad_front",
            exercise="shoulder",
            observed_variant="seated_press",
            criterion_levels=[None, 2, 2, 2],
            eligible_for_item_training=True,
        )
        self.assertFalse(three_pipeline.compatible(row))
        row["observed_variant"] = "standing_dumbbell_shoulder_press"
        self.assertTrue(three_pipeline.compatible(row))

    def test_loader_rechecks_variant_despite_existing_cache(self) -> None:
        """과거 캐시 목록에 있어도 제외 입력을 읽거나 특징화하지 않는다."""
        row = dict(
            id="seated",
            exercise="shoulder",
            observed_variant="seated_press",
            criterion_levels=[None, 2, 2, 2],
            eligible_for_item_training=True,
        )
        with patch("three_pipeline.records", return_value=[row]), patch(
            "three_pipeline.audit.read_json",
            return_value=dict(records=[dict(id="seated", paths={})]),
        ):
            self.assertEqual(three_pipeline.load(), [])
