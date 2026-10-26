"""최종 시험이 학습·선택 자료와 섞이면 실패하는지 검사한다."""

import copy
import unittest

from app.scoring.partition import audit_partitions


def row(identifier: str, person: str, source: str, role: str) -> dict:
    """서로 다른 원본 지문을 가진 검사용 메타데이터를 반환한다."""
    return dict(
        id=identifier,
        person=person,
        exercise="squats",
        sources=[source * 64],
        split=role,
    )


class PartitionTest(unittest.TestCase):
    """연결 성분과 과거 노출에 대한 실패 계약을 검사한다."""

    def test_disjoint_roles_pass_without_mutation(self) -> None:
        """역할이 다른 독립 그룹은 통과하며 원본 행을 바꾸지 않는다."""
        records = [
            row("a", "one", "a", "train"),
            row("b", "two", "b", "final_test"),
        ]
        before = copy.deepcopy(records)
        result = audit_partitions(records)
        self.assertEqual(records, before)
        self.assertEqual(result["role_counts"], {"train": 1, "final_test": 1})
        self.assertFalse(result["final_test_executed"])

    def test_transitive_overlap_is_rejected(self) -> None:
        """A의 사람을 공유하는 B의 다른 원본 C도 최종 시험에 넣지 않는다."""
        records = [
            row("a", "one", "a", "train"),
            row("b", "one", "b", "train"),
            row("c", "two", "b", "final_test"),
        ]
        with self.assertRaisesRegex(ValueError, "crosses roles"):
            audit_partitions(records)

    def test_person_shared_across_exercises_is_rejected(self) -> None:
        """종목만 다른 동일 참여자도 학습·최종 시험으로 나누지 않는다."""
        a, b = row("a", "one", "a", "train"), row("b", "one", "b", "final_test")
        b["exercise"] = "shoulder"
        with self.assertRaises(ValueError):
            audit_partitions([a, b])

    def test_reencoded_recording_is_rejected(self) -> None:
        """지문이 달라도 직접 확인한 재인코딩 묶음이 같으면 거부한다."""
        a, b = row("a", "one", "a", "train"), row("b", "two", "b", "validation")
        a["duplicate_groups"] = b["duplicate_groups"] = ["same-take"]
        with self.assertRaises(ValueError):
            audit_partitions([a, b])

    def test_historical_exposure_cannot_become_final_test(self) -> None:
        """이번 학습 목록에 없더라도 과거 노출 원본은 새 시험이 아니다."""
        with self.assertRaisesRegex(ValueError, "historical"):
            audit_partitions(
                [row("a", "one", "a", "final_test")],
                historical_sources=["a" * 64],
            )

    def test_unhashed_or_ambiguous_records_fail_closed(self) -> None:
        """누락 지문·추측한 역할·중복 ID는 분할 완료로 인정하지 않는다."""
        change: dict
        for change in (
            {"sources": []},
            {"sources": ["pending"]},
            {"split": "test"},
        ):
            record = row("a", "one", "a", "train")
            record.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit_partitions([record])
        a = row("a", "one", "a", "train")
        with self.assertRaises(ValueError):
            audit_partitions([a, a])

    def test_malformed_history_is_not_silently_ignored(self) -> None:
        """잘못된 과거 원본 지문으로 독립성 검사가 통과하지 않게 한다."""
        with self.assertRaisesRegex(ValueError, "Historical"):
            audit_partitions(
                [row("a", "one", "a", "final_test")],
                historical_sources=["A" * 64],
            )

    def test_parent_hash_links_cropped_video(self) -> None:
        """측면 잘라내기의 지문이 달라도 원본 지문을 연결하면 누출을 막는다."""
        crop = row("crop", "one", "b", "train")
        crop["sources"].append("a" * 64)
        with self.assertRaises(ValueError):
            audit_partitions(
                [crop, row("original", "other", "a", "final_test")]
            )

    def test_historical_duplicate_group_is_protected(self) -> None:
        """과거 사용한 재인코딩 묶음도 새 최종 시험으로 바꾸지 않는다."""
        record = row("new-encode", "new-id", "c", "final_test")
        record["duplicate_groups"] = ["known-take"]
        with self.assertRaisesRegex(ValueError, "historical"):
            audit_partitions(
                [record], historical_duplicate_groups=["known-take"]
            )
