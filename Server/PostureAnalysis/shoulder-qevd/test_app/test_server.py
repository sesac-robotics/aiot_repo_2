"""Regression checks for dataset filtering, label provenance and video seeking."""
import http.client
import json
import threading
import unittest
from html.parser import HTMLParser
from urllib.parse import urlencode

from server import make_server, parse_range, safe_path, SHOULDER


class RangeTests(unittest.TestCase):
    def test_open_and_suffix_ranges(self):
        self.assertEqual(parse_range(None, 100), (0, 99, False))
        self.assertEqual(parse_range("bytes=20-39", 100), (20, 39, True))
        self.assertEqual(parse_range("bytes=90-", 100), (90, 99, True))
        self.assertEqual(parse_range("bytes=-10", 100), (90, 99, True))
        self.assertEqual(parse_range("bytes=95-1000", 100), (95, 99, True))

    def test_invalid_ranges(self):
        for value in ["bytes=100-", "bytes=40-20", "bytes=-0", "bytes=-", "bytes=0-1,3-4", "potato"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_range(value, 100)

    def test_path_stays_inside_dataset(self):
        with self.assertRaises(FileNotFoundError):
            safe_path(SHOULDER, "../qevd-selected/manifests/short_clips.csv")


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path, method="GET", headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        data = response.read()
        result = (response.status, dict(response.getheaders()), data)
        connection.close()
        return result

    def api(self, path):
        status, _, body = self.request(path)
        self.assertEqual(status, 200, body)
        return json.loads(body)

    def test_all_five_exercises_and_counts(self):
        data = self.api("/api/stats")
        self.assertEqual(data["total"], 18752)
        self.assertEqual({r["id"]: r["count"] for r in data["exercises"]}, {
            "squats": 4308, "pushups": 4270, "elbow_plank": 6477, "ytw": 3597, "shoulder_press": 100})

    def test_drafts_do_not_include_pending_or_quarantine(self):
        data = self.api("/api/clips?dataset=shoulder&limit=100")
        self.assertEqual(data["total"], 100)
        self.assertTrue(all(r["status"] == "draft" for r in data["items"]))
        self.assertEqual(len(data["items"]), 100)

    def test_filters_and_pagination(self):
        data = self.api("/api/clips?dataset=shoulder&split=test&page=1&limit=5")
        self.assertEqual(data["total"], 16)
        self.assertEqual(len(data["items"]), 5)
        self.assertTrue(all(r["split"] == "test" for r in data["items"]))
        second = self.api("/api/clips?dataset=shoulder&split=test&page=2&limit=5")
        self.assertFalse({r["key"] for r in data["items"]} & {r["key"] for r in second["items"]})

    def test_ytw_variants_are_in_one_exercise_group(self):
        data = self.api("/api/clips?dataset=qevd&exercise=ytw")
        self.assertEqual(data["total"], 3597)
        self.assertEqual(len(data["labels"]), 14)

    def test_every_qevd_clip_has_traceable_draft_feedback(self):
        rows = [r for r in self.server.catalog.records if r["dataset"] == "qevd"]
        self.assertEqual(len(rows), 18652)
        for row in rows:
            self.assertTrue(row["feedbacks"], row["key"])
            self.assertEqual(row["feedback_status"], "draft")
            self.assertEqual(row["feedback_origin"], "assistant_authored_label_mapping")
            self.assertEqual(len(row["label_feedback_mapping"]), len(row["labels"]))
        self.assertEqual(sum(bool(r["original_feedbacks"]) for r in rows), 7107)

    def test_ytw_now_has_mapped_english_feedback(self):
        row = next(r for r in self.server.catalog.records if "standing ytw - arms too low" in r["labels"])
        self.assertIn("Raise your arms", row["feedbacks"][0])
        self.assertEqual(row["original_feedbacks"], [])

    def test_source_conflict_and_visibility_are_not_silent_corrections(self):
        rows = [r for r in self.server.catalog.records if r["dataset"] == "qevd"]
        row = next(r for r in rows if "squats - knees over toes" in r["labels"])
        entry = next(e for e in row["label_feedback_mapping"] if e["label"] == "squats - knees over toes")
        self.assertEqual(entry["category"], "source_conflict")
        self.assertIn("needs review", entry["feedback_en"])
        for row in rows:
            if any(e["category"] == "visibility" for e in row["label_feedback_mapping"]):
                self.assertTrue(all("camera" in f.lower() for f in row["feedbacks"]), row["key"])

    def test_feedback_and_evidence_match_selected_subject(self):
        data = self.api("/api/clips?" + urlencode({"dataset": "shoulder", "q": "subject_023", "evidence": "1"}))
        self.assertEqual(data["total"], 1)
        record = self.api("/api/clip?" + urlencode({"key": data["items"][0]["key"]}))
        self.assertEqual(record["source_quality"], "bad")
        self.assertIn("양팔", record["feedbacks_ko"][0])
        self.assertIn("uneven arm height", record["labels"][0])
        self.assertIsNotNone(record["evidence"])
        self.assertEqual(record["evidence"]["bad_time"], 11.5)

    def test_video_byte_range_support_for_browser_seeking(self):
        item = self.server.catalog.records[0]
        status, headers, body = self.request(item["media_url"], headers={"Range": "bytes=0-63"})
        self.assertEqual(status, 206)
        self.assertEqual(len(body), 64)
        self.assertEqual(headers["Accept-Ranges"], "bytes")
        self.assertEqual(headers["Content-Type"], "video/mp4")
        self.assertTrue(headers["Content-Range"].startswith("bytes 0-63/"))
        status, _, body = self.request(item["media_url"], method="HEAD")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"")

    def test_missing_clip_and_bad_input(self):
        self.assertEqual(self.request("/api/clip?key=no-such-id")[0], 404)
        self.assertEqual(self.request("/api/clips?page=invalid")[0], 400)
        self.assertEqual(self.request("/media/shoulder/../qevd-selected/manifests/short_clips.csv")[0], 404)

    def test_static_ui_and_cross_site_validation_rejection(self):
        status, _, body = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("운동 데이터셋 테스트".encode(), body)
        self.assertEqual(self.request("/api/validate", method="POST", headers={"Origin": "https://example.com"})[0], 403)

    def test_both_dataset_choices_are_real_select_options(self):
        class Options(HTMLParser):
            def __init__(self):
                super().__init__()
                self.in_dataset = False
                self.values = []
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "select":
                    self.in_dataset = attrs.get("id") == "dataset"
                if tag == "option" and self.in_dataset:
                    self.values.append(attrs.get("value"))
            def handle_endtag(self, tag):
                if tag == "select":
                    self.in_dataset = False
        _, _, body = self.request("/")
        parsed = Options()
        parsed.feed(body.decode())
        self.assertEqual(parsed.values, ["shoulder", "qevd", ""])


if __name__ == "__main__":
    unittest.main(verbosity=2)
