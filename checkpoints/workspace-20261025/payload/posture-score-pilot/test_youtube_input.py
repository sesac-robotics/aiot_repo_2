"""유튜브 입력 경계와 기록 중복·보존·삭제 동작을 검증한다."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from saved_sources import SourceLibrary
from youtube_input import validate_request


class YoutubeInputTests(unittest.TestCase):
    """외부 접속 없이 URL 허용 범위와 시간 경계를 검사한다."""

    def test_video_urls_normalized_without_playlist_or_time_query(self):
        """영상 ID만 유지해 외부 주소·재생목록 전달을 차단한다."""
        for url in (
            "https://www.youtube.com/watch?v=JyCG_5l3XLk&list=abc&t=30",
            "https://youtu.be/JyCG_5l3XLk?t=30",
            "https://m.youtube.com/shorts/JyCG_5l3XLk",
        ):
            with self.subTest(url=url):
                result = validate_request({"url": url, "exercise": "pushup"})
                self.assertEqual(
                    result["url"],
                    "https://www.youtube.com/watch?v=JyCG_5l3XLk",
                )

    def test_untrusted_hosts_and_non_video_links_rejected(self):
        """내부 주소·유사 도메인·인증정보·재생목록은 거절한다."""
        for url in (
            "http://127.0.0.1/secret",
            "file:///etc/passwd",
            "https://youtube.com.example.org/watch?v=JyCG_5l3XLk",
            "https://youtube.com@127.0.0.1/watch?v=JyCG_5l3XLk",
            "https://www.youtube.com:8080/watch?v=JyCG_5l3XLk",
            "https://www.youtube.com/playlist?list=abc",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_request({"url": url, "exercise": "squat"})

    def test_invalid_time_and_exercise_rejected(self):
        """무한·음수·범위 초과와 미지원 운동을 거절한다."""
        base = {"url": "https://youtu.be/JyCG_5l3XLk", "exercise": "pushup"}
        for patch in (
            {"start_seconds": -1},
            {"start_seconds": "NaN"},
            {"duration_seconds": 121},
            {"duration_seconds": 0},
            {"duration_seconds": "inf"},
            {"exercise": "plank"},
        ):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_request({**base, **patch})


class SourceLibraryTests(unittest.TestCase):
    """재시작 후 기록 보존·구간별 구분·삭제 경합을 검사한다."""

    def test_deduplication_persistence_and_deleted_completion(self):
        """중복 입력은 합치고 구간 차이는 보존하며 삭제를 취소하지 않는다."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = SourceLibrary(root)
            request = validate_request(
                {
                    "url": "https://youtu.be/JyCG_5l3XLk",
                    "exercise": "pushup",
                }
            )
            ident = library.remember(request)
            self.assertEqual(ident, library.remember(request))
            library.remember({**request, "start_seconds": 10})
            library.complete(ident, "Pushup example", "a" * 32)
            restored = SourceLibrary(root)
            self.assertEqual(len(restored.list()), 2)
            stored = next(x for x in restored.list() if x["id"] == ident)
            self.assertEqual(stored["last_job_id"], "a" * 32)
            self.assertTrue(restored.delete(ident))
            restored.complete(ident, "Late completion", "b" * 32)
            self.assertFalse(restored.delete(ident))
            self.assertEqual(len(SourceLibrary(root).list()), 1)

    def test_removed_default_not_reseeded(self):
        """기본 예제도 목록 삭제 후 재시작하면 다시 생성되지 않는다."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = "c" * 32
            folder = root / "jobs" / job
            folder.mkdir(parents=True)
            (root / "youtube-demo-jobs.json").write_text(
                json.dumps({"squat": job})
            )
            (folder / "result.json").write_text(
                json.dumps(
                    {
                        "source": {
                            "url": "https://youtu.be/m0GcZ24pK6k",
                            "title": "Squat",
                            "original_start_seconds": 5,
                            "original_end_seconds": 18,
                        }
                    }
                )
            )
            library = SourceLibrary(root)
            ident = library.list()[0]["id"]
            library.delete(ident)
            self.assertEqual(SourceLibrary(root).list(), [])
            self.assertTrue((folder / "result.json").is_file())


class YoutubeJobTests(unittest.TestCase):
    """다운로드 실패와 최종 영상 합성 전 완료 노출을 검사한다."""

    def test_download_failure_releases_analysis_slot(self):
        """네트워크 오류 후 다음 분석을 받을 수 있어야 한다."""
        import server

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "jobs" / "test").mkdir(parents=True)
            lock = __import__("threading").Lock()
            lock.acquire()
            with patch.object(server, "ROOT", root), patch.object(
                server, "LOCK", lock
            ), patch.object(server, "STATES", {}), patch.object(
                server, "download_clip", side_effect=ValueError("다운로드 실패")
            ):
                server.run_job("test", "squat", {})
                self.assertEqual(server.STATES["test"]["state"], "failed")
                self.assertFalse(lock.locked())

    def test_done_only_after_score_video_rendered(self):
        """기존 분석기의 done 콜백은 최종 합성이 끝날 때까지 숨긴다."""
        import server

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "jobs" / "test").mkdir(parents=True)
            lock = __import__("threading").Lock()
            lock.acquire()
            result = {"exercise": "pushup"}

            def fake_analyze(source, folder, exercise, progress):
                progress({"state": "done", "result": result})
                self.assertNotEqual(
                    server.STATES.get("test", {}).get("state"), "done"
                )
                return result

            def fake_render(folder):
                self.assertEqual(server.STATES["test"]["state"], "rendering")

            with patch.object(server, "ROOT", root), patch.object(
                server, "LOCK", lock
            ), patch.object(server, "STATES", {}), patch.object(
                server,
                "download_clip",
                return_value={"title": "Example", "channel": "Example"},
            ), patch.object(
                server, "analyze", side_effect=fake_analyze
            ), patch.object(
                server, "render", side_effect=fake_render
            ):
                server.run_job("test", "pushup", {})
                self.assertEqual(server.STATES["test"]["state"], "done")
                self.assertFalse(lock.locked())


if __name__ == "__main__":
    unittest.main()
