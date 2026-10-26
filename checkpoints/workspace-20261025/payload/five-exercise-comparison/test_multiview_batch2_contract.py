"""두 번째 스쿼트 배치의 불완전 구간·관측·기존 결과 격리를 검사한다."""

import pathlib
import types
import unittest

import numpy as np
from unittest.mock import patch

import multiview_squat_trial
from multiview_squat_batch2_train import (
    configured_script,
    load_batch,
    fit_models,
    mask_reviewed_frames,
)
from recovery_train import component_groups


class Batch2ContractTests(unittest.TestCase):
    """가림·불완전 반복을 총점 가능 자료로 만드는 회귀를 차단한다."""

    def test_configuration_does_not_modify_existing_module(self):
        """배치 경로 설정이 보존된 첫 배치 모듈에 전파되지 않음을 확인한다."""
        prior = multiview_squat_trial.OUT
        loaded = configured_script(
            "multiview_squat_trial.py", pathlib.Path("/tmp/batch2-test")
        )
        self.assertEqual(multiview_squat_trial.OUT, prior)
        self.assertEqual(loaded.OUT, pathlib.Path("/tmp/batch2-test"))

    def test_incomplete_attempt_cannot_get_total_permission(self):
        """네 항목 등급이 있어도 시작이 잘린 반복의 총점은 허용하지 않는다."""
        row = dict(
            id="a", notes=[dict(item="descent_range", eligible=True, holds=[])]
        )
        module = types.SimpleNamespace(load_new=lambda receipts: [row])
        ann = dict(
            id="a",
            attempt_complete=False,
            eligible_for_total_score_training=False,
        )
        with patch(
            "multiview_squat_batch2_train.configured_script",
            return_value=module,
        ), patch(
            "multiview_squat_batch2_train.audit.read_json",
            side_effect=[
                dict(records=[]),
                [ann],
                dict(records=[dict(id="a", view=v) for v in ("side", "front")]),
            ],
        ):
            result = load_batch(pathlib.Path("/tmp/unused"))[0]
        self.assertFalse(result["whole_attempt_observed"])
        self.assertFalse(result["reference_total_allowed"])

    def test_hidden_bottom_cannot_be_restored_by_confident_landmarks(self):
        """흐림으로 가린 최저점을 관절 신뢰도만으로 복구하지 않는다."""
        row = dict(
            id="a",
            notes=[
                dict(item="descent_range", eligible=True, holds=[]),
                dict(item="trunk_balance", eligible=True, holds=[]),
            ],
        )
        module = types.SimpleNamespace(load_new=lambda receipts: [row])
        ann = dict(
            id="a",
            attempt_complete=True,
            eligible_for_total_score_training=False,
            manual_unobservable_items=["descent_range"],
        )
        with patch(
            "multiview_squat_batch2_train.configured_script",
            return_value=module,
        ), patch(
            "multiview_squat_batch2_train.audit.read_json",
            side_effect=[
                dict(records=[]),
                [ann],
                dict(records=[dict(id="a", view=v) for v in ("side", "front")]),
            ],
        ):
            result = load_batch(pathlib.Path("/tmp/unused"))[0]
        self.assertFalse(result["notes"][0]["eligible"])
        self.assertTrue(result["notes"][1]["eligible"])
        self.assertFalse(result["reference_total_allowed"])

    def test_same_person_in_different_views_cannot_cross_roles(self):
        """서로 다른 시점·원본이어도 같은 참여자 메타 그룹 누출을 막는다."""
        rows = [
            dict(person="subject_012", sources={"side-a"}, split="train"),
            dict(person="subject_012", sources={"front-b"}, split="validation"),
        ]
        with self.assertRaises(ValueError):
            component_groups(rows)

    def test_heel_boundary_can_learn_from_two_error_groups(self):
        """무릎 각도가 같아도 뒤꿈치 오류의 직접 등급을 보정할 수 있다."""
        rows = []
        for person, grade, heel in (
            ("normal_a", 2, 0.01),
            ("normal_b", 2, 0.01),
            ("error_a", 0, 0.05),
            ("error_b", 0, 0.05),
        ):
            note = dict(
                item="knee_foot",
                bank="side",
                eligible=True,
                feature=[0] * 7 + [1, heel],
                measurement=0.0,
                higher_is_better=False,
            )
            rows.append(
                dict(
                    exercise="squats",
                    person=person,
                    split="train",
                    criterion_ids=["knee_foot"],
                    levels=[grade],
                    notes=[note],
                )
            )
        head = fit_models(rows)["squats", "knee_foot", "side"]
        self.assertEqual(len(head["boundary"]), 4)
        self.assertLess(
            head["heel_calibration"]["training_weighted_MAE_after"],
            head["heel_calibration"]["training_weighted_MAE_before"],
        )
        self.assertTrue(head["heel_calibration"]["training_only"])

    def test_manual_frame_mask_preserves_other_observations(self):
        """제외 표본 범위만 NaN으로 바꾸고 원본과 나머지 관측은 보존한다."""
        source = np.ones((5, 33, 2))
        result = mask_reviewed_frames(
            source, np.array([0, 2, 4, 6, 8]), [[4, 6]]
        )
        self.assertTrue(np.isnan(result[2:4]).all())
        self.assertTrue(np.isfinite(result[[0, 1, 4]]).all())
        self.assertTrue(np.isfinite(source).all())

    def test_manual_frame_mask_rejects_reversed_range(self):
        """제외 구간이 역전되면 조용히 넘어가지 않는다."""
        with self.assertRaises(ValueError):
            mask_reviewed_frames(
                np.ones((2, 33, 2)), np.array([1, 2]), [[2, 1]]
            )

    def test_validation_cannot_enter_boundary_selection(self):
        """검증 행이 경계 선택 함수에 전달되면 즉시 실패한다."""
        with self.assertRaises(ValueError):
            fit_models([dict(split="validation")])


if __name__ == "__main__":
    unittest.main()
