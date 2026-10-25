"""완료 전 파일 거부와 ZIP의 비실행 인수 검사를 확인한다."""

import pathlib
import tempfile
import unittest
import zipfile
from aihub_intake import inspect_archive


class IntakeTests(unittest.TestCase):
    """원본 압축 해제 없이 경로·모델 후보를 확인한다."""

    def test_partial_download_rejected(self):
        """다운로드 중인 파일은 내용 읽기 전에 거부해야 한다."""
        with self.assertRaises(ValueError):
            inspect_archive(pathlib.Path("missing.crdownload"))

    def test_metadata_only_detects_unsafe_path(self):
        """외부 경로 멤버를 발견해도 압축을 풀거나 코드를 실행하지 않는다."""
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "model.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "../outside.py", "raise RuntimeError('do not execute')"
                )
                archive.writestr("model.pth", b"not a loaded checkpoint")
                archive.writestr("README.txt", "입출력 확인 필요")
            report = inspect_archive(path)
            self.assertEqual(report["unsafe_paths"], ["../outside.py"])
            self.assertEqual(report["model_candidates"], ["model.pth"])
            self.assertFalse((path.parent / "model.pth").exists())
            self.assertIn("입출력", report["descriptions"][0]["text"])


if __name__ == "__main__":
    unittest.main()
