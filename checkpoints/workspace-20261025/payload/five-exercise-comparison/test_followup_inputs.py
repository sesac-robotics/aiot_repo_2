"""추가 실험의 원본 그룹과 항목 관측 계약 회귀를 검사한다."""

import unittest

import numpy as np

from plank_partial_item_trial import source_identity
from w_item_learning import features


class FollowupInputTests(unittest.TestCase):
    """원본 null과 좌표계 변화가 학습·검증 계약을 깨지 않는지 검사한다."""

    def test_explicit_null_uses_real_input_source(self) -> None:
        """null을 공유하는 무관한 원본이 같은 그룹으로 묶이지 않는다."""
        a = source_identity(
            dict(source_original_sha256=None, video_sha256="a" * 64)
        )
        b = source_identity(
            dict(source_original_sha256=None, video_sha256="b" * 64)
        )
        self.assertNotEqual(a, b)

    def test_original_source_has_priority(self) -> None:
        """같은 원본의 다른 편집본은 원본 SHA로 계속 묶는다."""
        a = source_identity(
            dict(source_original_sha256="a" * 64, video_sha256="b" * 64)
        )
        b = source_identity(
            dict(source_original_sha256="a" * 64, video_sha256="c" * 64)
        )
        self.assertEqual(a, b)

    def test_missing_identity_is_an_error(self) -> None:
        """미상 그룹을 임의 ID나 null로 채우지 않고 중단한다."""
        with self.assertRaises(ValueError):
            source_identity({})

    def test_w_geometry_ignores_fixed_translation_and_scale(self) -> None:
        """동일 모양의 화면 이동·확대가 W 학습 특징을 바꾸지 않는다."""
        seq = np.full((5, 33, 2), np.nan)
        for j, xy in [
            (11, (20, 0)),
            (12, (-20, 0)),
            (23, (20, 100)),
            (24, (-20, 100)),
            (13, (70, 0)),
            (14, (-70, 0)),
            (15, (70, -50)),
            (16, (-70, -50)),
        ]:
            seq[:, j] = xy
        np.testing.assert_allclose(
            features(seq, 0, 4),
            features(seq * 2 + [400, 300], 0, 4),
            atol=1e-10,
        )

    def test_w_required_wrist_occlusion_is_unknown(self) -> None:
        """특징을 만들 수 없는 W 관측은 값 대체로 평가 가능하게 만들지 않는다."""
        seq = np.full((5, 33, 2), np.nan)
        self.assertIsNone(features(seq, 0, 4))
