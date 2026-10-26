"""검토 라벨의 출처 보존ㆍ승인 조건ㆍCSV 제외 규칙을 검사한다."""

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from review import export_csv, load_review, save_review


class ReviewTests(unittest.TestCase):
    """미검토 초안이 검토 완료 라벨로 섞이지 않는지 확인한다."""

    def setUp(self):
        """0~100점 단위의 반복 결과가 있는 임시 작업 폴더를 만든다."""
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.result = {
            "exercise": "squat",
            "rules": {"version": "test"},
            "repetitions": [
                {"number": 1, "start": 0, "end": 3, "score": 80},
                {"number": 2, "start": 3, "end": 6, "score": 70},
            ],
        }
        (self.folder / "result.json").write_text(json.dumps(self.result))
        self.payload = dict(
            number=1,
            status="accepted",
            subject_id="P001",
            side_confirmed=True,
            revision=0,
        )

    def tearDown(self):
        """검사 전용 임시 폴더를 정리한다."""
        self.temp.cleanup()

    def rows(self, reviewed=False):
        """CSV 바이트를 행 사전 목록으로 읽어 반환한다."""
        return list(
            csv.DictReader(
                io.StringIO(
                    export_csv(self.folder, reviewed).decode("utf-8-sig")
                )
            )
        )

    def test_unreviewed_not_training_labels(self):
        """미검토 행은 검토 CSV에 없고 원래 초안 CSV에만 남는다."""
        self.assertEqual(self.rows(True), [])
        self.assertEqual(self.rows()[0]["reviewed_score"], "")

    def test_accept_keeps_original(self):
        """사람이 승인해도 원본 점수는 보존되고 검토 정보만 추가된다."""
        before = (self.folder / "result.json").read_bytes()
        save_review(self.folder, self.payload)
        self.assertEqual((self.folder / "result.json").read_bytes(), before)
        rows = self.rows(True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reviewed_score"], "80")
        self.assertEqual(rows[0]["expert_validated"], "False")

    def test_validation_and_revision(self):
        """구도 미확인ㆍ비정상 수정 점수ㆍ동시 수정 충돌을 거절한다."""
        with self.assertRaises(ValueError):
            save_review(self.folder, dict(self.payload, side_confirmed=False))
        with self.assertRaises(ValueError):
            save_review(
                self.folder,
                dict(self.payload, status="corrected", score=float("nan")),
            )
        save_review(self.folder, self.payload)
        with self.assertRaises(RuntimeError):
            save_review(self.folder, self.payload)
        save_review(
            self.folder, dict(self.payload, status="rejected", revision=1)
        )
        self.assertEqual(self.rows(True), [])
        self.assertEqual(len(load_review(self.folder)["history"]), 2)

    def test_result_change_invalidates_labels(self):
        """원본 분석이 바뀌면 이전 검토를 새 결과에 자동 적용하지 않는다."""
        save_review(self.folder, self.payload)
        self.result["repetitions"][0]["score"] = 10
        (self.folder / "result.json").write_text(json.dumps(self.result))
        self.assertEqual(self.rows(True), [])


if __name__ == "__main__":
    unittest.main()
