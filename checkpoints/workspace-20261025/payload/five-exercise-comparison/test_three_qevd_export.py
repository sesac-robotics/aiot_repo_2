"""QEVD 기본 JSON 연결에서 분할 누출·참여자 오연결·구간 유실을 검사한다."""

import copy
import unittest

import three_protocol
import three_qevd_export


class QevdExportTest(unittest.TestCase):
    """별도 점수 주석이 원본 기본 필드와 안전하게 연결되는지 확인한다."""

    def setUp(self) -> None:
        """원본과 개발 분할이 다른 비전문가 점수 초안을 준비한다."""
        self.original = dict(
            video_path="./00000001.mp4",
            labels=["provider error"],
            labels_descriptive=["original description"],
            split="train",
        )
        self.workers = [dict(video_path="./00000001.mp4", worker_id=17)]
        self.row = three_protocol.convert(
            dict(
                id="case",
                video_path="/local/00000001.mp4",
                video_sha256="a" * 64,
                person="qevd_17",
                split="validation",
                exercise="pushups",
                labels=["provider error"],
                labels_descriptive=["original description"],
                criterion_ids=["a", "b", "c", "d"],
                criterion_weights=[40, 30, 20, 10],
                criterion_levels=[2, 1, None, 2],
                score=None,
                rubric_version="test",
                segment=dict(start_frame=0, end_frame=30, fps=30),
            ),
            dataset="QEVD",
        )

    def test_native_values_and_local_validation_are_both_preserved(
        self,
    ) -> None:
        """원본 문자열·경로·train과 로컬 validation을 서로 덮어쓰지 않는다."""
        result = three_qevd_export.build_bundle(
            [self.row], [self.original], self.workers
        )
        self.assertEqual(result["fine_grained_labels"], [self.original])
        self.assertEqual(result["development_splits"][0]["split"], "validation")
        self.assertEqual(
            result["score_annotations"][0]["assessment"]["criterion_levels"],
            [2, 1, None, 2],
        )
        self.assertIsNone(result["score_annotations"][0]["assessment"]["score"])
        self.assertEqual(
            result["media_manifest"][0]["local_paths"], ["/local/00000001.mp4"]
        )

    def test_original_worker_is_authoritative(self) -> None:
        """파일명 추측 참여자가 실제 원본 참여자와 다르면 내보내기를 막는다."""
        self.row["worker_id"] = 18
        with self.assertRaises(ValueError):
            three_qevd_export.build_bundle(
                [self.row], [self.original], self.workers
            )

    def test_native_test_cannot_become_development(self) -> None:
        """원본 test를 로컬 validation으로 바꾸어 학습 개발에 섞지 않는다."""
        self.original["split"] = "test"
        with self.assertRaises(ValueError):
            three_qevd_export.build_bundle(
                [self.row], [self.original], self.workers
            )

    def test_multiple_segments_keep_one_video_and_two_independent_annotations(
        self,
    ) -> None:
        """한 긴 영상에 정상·오류 구간이 있어도 구간 주석을 합치지 않는다."""
        row = copy.deepcopy(self.row)
        row["provenance"]["dataset"] = "External"
        row["worker_id"] = 1000001
        other = copy.deepcopy(row)
        other["id"] = "case2"
        other["labels"] = ["different provider phrase"]
        other["segment"] = dict(start_frame=60, end_frame=90, fps=30)
        result = three_qevd_export.build_bundle([row, other], [], [])
        self.assertEqual(len(result["fine_grained_labels"]), 1)
        self.assertEqual(len(result["score_annotations"]), 2)
        self.assertEqual(
            result["fine_grained_labels"][0]["labels"],
            ["provider error", "different provider phrase"],
        )
        self.assertNotEqual(
            result["score_annotations"][0]["segment"],
            result["score_annotations"][1]["segment"],
        )
