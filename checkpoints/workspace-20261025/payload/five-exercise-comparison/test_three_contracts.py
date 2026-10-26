"""3종 JSON·관측·총점 계약에서 실제 오류를 막는 경계를 검사한다."""

import copy
import unittest

import numpy as np

import three_items
import three_protocol
import three_training


class ThreeContractsTest(unittest.TestCase):
    """출처 문자열과 독립 라벨, 구도·관측·배점의 분리를 확인한다."""

    def setUp(self) -> None:
        """학습·원본 파일 없이 실행할 최소 QEVD 호환 예제를 준비한다."""
        self.row = dict(
            id="synthetic",
            exercise="pushups",
            video_path="./synthetic.mp4",
            video_sha256="a" * 64,
            person="qevd_17",
            split="train",
            labels=["provider bad"],
            labels_descriptive=["provider text"],
            criterion_ids=["a", "b", "c", "d"],
            criterion_weights=[40, 30, 20, 10],
            criterion_levels=[2, 1, 2, 2],
            score=85,
            rubric_version="synthetic-only",
            label_source="manual_draft",
        )

    def test_qevd_strings_and_scores_remain_independent(self) -> None:
        """제공자 오류 문자열이 있어도 점수는 입력의 직접 등급을 보존한다."""
        original = copy.deepcopy(self.row)
        result = three_protocol.convert(self.row, dataset="QEVD")
        self.assertEqual(result["labels"], ["provider bad"])
        self.assertEqual(result["assessment"]["score"], 85)
        self.assertEqual(result["worker_id"], 17)
        self.assertEqual(self.row, original)

    def test_missing_grade_cannot_have_a_total(self) -> None:
        """항목 결측을 남긴 채 총점 85를 유지하면 계약 위반이다."""
        self.row["criterion_levels"][0] = None
        with self.assertRaises(ValueError):
            three_protocol.convert(self.row, dataset="QEVD")

    def test_external_person_is_not_a_provider_worker(self) -> None:
        """외부 참여자를 QEVD 원래 숫자 참여자라고 꾸미지 않는다."""
        self.row["person"] = "other_17"
        result = three_protocol.convert(self.row, dataset="MultiView")
        self.assertIsNone(result["worker_id"])

    def test_partial_scores_are_not_renormalized(self) -> None:
        """세 항목 만점만 관측해도 전신 만점을 반환하지 않는다."""
        self.assertIsNone(three_items.total([None, 2, 2, 2], [40, 30, 20, 10]))

    def test_shoulders_only_do_not_make_a_full_score(self) -> None:
        """어깨·골반만 보일 때 몸통 항목 외 팔 항목은 평가하지 않는다."""
        sequence = np.full((30, 33, 2), np.nan)
        for joint, xy in [
            (11, (90, 100)),
            (12, (130, 100)),
            (23, (90, 200)),
            (24, (130, 200)),
        ]:
            sequence[:, joint] = xy
        notes = three_items.describe("shoulder", sequence, None)
        self.assertEqual(
            [n["eligible"] for n in notes], [False, False, False, True]
        )

    def test_top_occlusion_cannot_be_hidden_by_average_visibility(self) -> None:
        """평균 관측률 80%여도 상단 손목이 모두 사라지면 도달을 보류한다."""
        sequence = np.full((30, 33, 2), np.nan)
        for joint, xy in [
            (11, (90, 100)),
            (12, (130, 100)),
            (13, (70, 110)),
            (14, (150, 110)),
            (15, (70, 80)),
            (16, (150, 80)),
            (23, (90, 200)),
            (24, (130, 200)),
        ]:
            sequence[:, joint] = xy
        sequence[10:16, 13] = (85, 50)
        sequence[10:16, 14] = (135, 50)
        sequence[10:16, [15, 16]] = np.nan
        notes = three_items.describe("shoulder", sequence, None)
        self.assertFalse(notes[0]["eligible"])
        self.assertTrue(notes[1]["eligible"])
        self.assertFalse(notes[2]["eligible"])

    def test_squat_front_only_keeps_trunk_but_no_total(self) -> None:
        """동기 측면이 없으면 정면 몸통만 평가하고 총점은 만들지 않는다."""
        sequence = np.full((30, 33, 2), np.nan)
        for joint, xy in [
            (11, (90, 100)),
            (12, (130, 100)),
            (23, (90, 200)),
            (24, (130, 200)),
            (25, (90, 260)),
            (26, (130, 260)),
            (27, (90, 320)),
            (28, (130, 320)),
        ]:
            sequence[:, joint] = xy
        notes = three_items.describe("squats", None, sequence)
        self.assertEqual(
            [n["eligible"] for n in notes], [False, False, False, True]
        )

    def test_rule_boundary_order_is_checked(self) -> None:
        """높을수록 좋은 기준에서 만점 경계가 더 낮으면 거부한다."""
        note = dict(
            eligible=True,
            measurement=170,
            higher_is_better=True,
            item="trunk_line",
        )
        with self.assertRaises(ValueError):
            three_training.rule_grade(note, (150, 160))

    def test_border_mask_does_not_remove_unrelated_trunk_information(
        self,
    ) -> None:
        """손목이 경계 밖이어도 필요한 몸통 정보까지 지우지 않는다."""
        sequence = np.full((30, 33, 2), np.nan)
        for joint, xy in [
            (11, (100, 100)),
            (12, (140, 100)),
            (23, (100, 200)),
            (24, (140, 200)),
            (15, (100, 1)),
        ]:
            sequence[:, joint] = xy
        masked = three_items.mask_image_border(sequence, 400, 400)
        self.assertTrue(np.isnan(masked[:, 15]).all())
        self.assertTrue(
            three_items.describe("shoulder", masked, None)[3]["eligible"]
        )
        self.assertTrue(np.isfinite(sequence[:, 15]).all())

    def test_one_visible_stationary_arm_is_an_error_not_missing(self) -> None:
        """한 팔만 움직이는 관측된 오류를 관절 결측으로 버리지 않는다."""
        sequence = np.full((30, 33, 2), np.nan)
        for joint, xy in [
            (11, (90, 100)),
            (12, (130, 100)),
            (13, (70, 110)),
            (14, (150, 110)),
            (15, (70, 80)),
            (16, (150, 80)),
            (23, (90, 200)),
            (24, (130, 200)),
        ]:
            sequence[:, joint] = xy
        sequence[:, 15, 1] = 80 - 70 * np.sin(np.linspace(0, np.pi, 30))
        note = three_items.describe("shoulder", sequence, None)[2]
        self.assertTrue(note["eligible"])
        self.assertGreater(note["measurement"], 0.8)

    def test_return_endpoint_does_not_require_five_final_samples(self) -> None:
        """빠른 복귀의 마지막 두 표본이 보이면 항목 관측을 허용한다."""
        sequence = np.full((10, 33, 2), np.nan)
        sequence[:, 11] = [100, 100]
        sequence[:, 13] = [100, 150]
        sequence[-2:, 15] = [100, 200]
        note = three_items.describe("pushups", sequence, None)[2]
        self.assertTrue(note["eligible"])
        self.assertAlmostEqual(note["measurement"], 180)

    def test_bent_elbows_cannot_get_full_overhead_reach(self) -> None:
        """상완만 머리 위에 있어도 팔꿈치가 크게 굽으면 도달 만점이 아니다."""
        sequence = np.full((30, 33, 2), np.nan)
        for joint, xy in [
            (11, (90, 100)),
            (12, (130, 100)),
            (13, (90, 50)),
            (14, (130, 50)),
            (15, (40, 50)),
            (16, (180, 50)),
            (23, (90, 200)),
            (24, (130, 200)),
        ]:
            sequence[:, joint] = xy
        note = three_items.describe("shoulder", sequence, None)[0]
        self.assertTrue(note["eligible"])
        self.assertAlmostEqual(note["measurement"], 90)
        self.assertEqual(three_training.rule_grade(note, (150, 125)), 0)

    def test_raised_bent_arms_are_not_a_completed_lower_return(self) -> None:
        """팔꿈치만 굽히고 상완을 위에 남긴 경우 하단 복귀 만점을 막는다."""
        sequence = np.full((30, 33, 2), np.nan)
        for joint, xy in [
            (11, (90, 100)),
            (12, (130, 100)),
            (13, (90, 50)),
            (14, (130, 50)),
            (15, (40, 50)),
            (16, (180, 50)),
            (23, (90, 200)),
            (24, (130, 200)),
        ]:
            sequence[:, joint] = xy
        note = three_items.describe("shoulder", sequence, None)[1]
        self.assertTrue(note["eligible"])
        self.assertAlmostEqual(note["measurement"], 180)
        self.assertEqual(three_training.rule_grade(note, (110, 140)), 0)

    def test_oblique_view_does_not_receive_a_frontal_trunk_score(self) -> None:
        """사선에서 팔이 보여도 정면 몸통 항목을 임의 만점으로 채우지 않는다."""
        notes = [dict(eligible=True, holds=[]) for _ in range(4)]
        result = three_items.apply_view_policy(
            "shoulder", notes, ["front_oblique"]
        )
        self.assertEqual(
            [r["eligible"] for r in result], [True, True, True, False]
        )
        self.assertTrue(notes[3]["eligible"])

    def test_small_but_clear_heel_lift_uses_torso_ratio_units(self) -> None:
        """몸통 길이의 10%인 뒤꿈치 상승을 15cm 같은 큰 임계값으로 놓치지 않는다."""
        note = dict(
            eligible=True,
            measurement=0,
            higher_is_better=False,
            item="knee_foot",
            feature=[0] * 8 + [0.1],
        )
        self.assertEqual(
            three_training.rule_grade(note, (0.25, 0.5, 0.025, 0.08)), 0
        )

    def test_medial_knee_collapse_is_detected_even_if_present_at_start(
        self,
    ) -> None:
        """처음부터 안쪽으로 모인 무릎을 초기 자세 차감으로 정상화하지 않는다."""
        side = np.full((30, 33, 2), np.nan)
        for joint, xy in [
            (11, (100, 100)),
            (23, (100, 200)),
            (29, (100, 300)),
            (31, (130, 300)),
        ]:
            side[:, joint] = xy
        front = np.full((30, 33, 2), np.nan)
        for joint, xy in [
            (11, (90, 100)),
            (12, (130, 100)),
            (23, (90, 200)),
            (24, (130, 200)),
            (25, (110, 250)),
            (26, (110, 250)),
            (27, (90, 300)),
            (28, (130, 300)),
        ]:
            front[:, joint] = xy
        note = three_items.describe("squats", side, front)[2]
        self.assertTrue(note["eligible"])
        self.assertGreater(note["measurement"], 0.2)
        self.assertEqual(
            three_training.rule_grade(note, (0.1, 0.2, 0.025, 0.08)), 0
        )
