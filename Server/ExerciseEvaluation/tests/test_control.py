"""실제 카메라 없이 모드 격리·세션 경계·자동 대체·HTTP 접근을 검사합니다."""

import ast
import json
import threading
import time
import unittest
import urllib.error
import urllib.request

from Server.ExerciseEvaluation.camera_sources import ChoiceCamera
from Server.ExerciseEvaluation.contracts import ObservedCadence, validate
from Server.ExerciseEvaluation.coordinator import Coordinator
from Server.ExerciseEvaluation.paths import REPO, SCORE_LIVE
from Server.ExerciseEvaluation.server import make_server
from Server.ExerciseEvaluation.worker import final_response


class FakeCamera:
    """실장치 접근 없이 연결 대기 상태만 제공합니다."""

    def __init__(self, options):
        self.selected = options["camera"]
        self.closed = False

    def start(self):
        pass

    def snapshot(self):
        return 0, 1, None, False, "fixture offline", 0

    def stop(self):
        self.closed = True


class FakeWorker:
    def __init__(self, kind, exercise, python, callback):
        self.kind = kind
        self.closed = False

    def close(self):
        self.closed = True


class ControlTests(unittest.TestCase):
    def test_stop_after_source_switch_does_not_relabel_old_results(self):
        class Adapter:
            def flush(self):
                return [{"number": 7}]

            def snapshot(self):
                return {"movement": {"count": 7}}

        meta = {"generation": 1, "epoch": 2, "timestamp": 4}
        self.assertNotIn(
            "repetitions", final_response(Adapter(), "score", 1, meta)
        )
        self.assertEqual(
            final_response(Adapter(), "score", 2, meta)["repetitions"],
            [{"number": 7}],
        )

    def test_low_rate_cannot_fill_a_continuous_six_hz_window(self):
        clock = ObservedCadence(6)
        run = longest = 0
        for i in range(200):
            continuous = clock.observe(i / 4)
            if continuous is not None:
                run = run + 1 if continuous else 1
                longest = max(longest, run)
        self.assertLess(longest, 32)
        clock = ObservedCadence(6)
        selected = [clock.observe(i / 10) for i in range(60)]
        self.assertGreaterEqual(sum(v is True for v in selected), 32)
        self.assertFalse(any(v is False for v in selected))

    def test_copied_score_functions_preserve_frozen_calculation(self):
        old = ast.parse((SCORE_LIVE / "server.py").read_text())
        new = ast.parse(
            (REPO / "Server/ExerciseEvaluation/legacy_score.py").read_text()
        )
        for node in new.body:
            if isinstance(node, ast.FunctionDef):
                original = next(
                    n
                    for n in old.body
                    if isinstance(n, ast.FunctionDef) and n.name == node.name
                )
                self.assertEqual(
                    ast.dump(node, include_attributes=False),
                    ast.dump(original, include_attributes=False),
                )

    def make_coordinator(self, factory=FakeWorker, fixed=None):
        return Coordinator(
            {"correction": "unused", "score": "unused"},
            fixed_mode=fixed,
            camera_factory=FakeCamera,
            worker_factory=factory,
        )

    def test_only_selected_engines_are_started_and_mode_switch_closes_old(self):
        c = self.make_coordinator()
        try:
            for mode, expected in [
                ("score", {"score"}),
                ("correction", {"correction"}),
            ]:
                old = list(c.workers.values())
                generation = c.generation
                c.start({"mode": mode})
                self.assertEqual(set(c.workers), expected)
                self.assertTrue(all(w.closed for w in old))
                self.assertEqual(c.generation, generation + 1)
                for kind in {"score", "correction"} - expected:
                    self.assertEqual(
                        c.snapshot()[kind]["status"], "not_requested"
                    )
        finally:
            workers = list(c.workers.values())
            camera = c.camera
            c.stop()
            self.assertTrue(camera.closed)
            self.assertTrue(all(w.closed for w in workers))

    def test_failed_mode_can_be_replaced_by_another_single_mode(self):
        def factory(kind, *args):
            if kind == "correction":
                raise FileNotFoundError("missing runtime")
            return FakeWorker(kind, *args)

        c = self.make_coordinator(factory)
        try:
            c.start({"mode": "correction"})
            self.assertEqual(set(c.workers), set())
            self.assertEqual(
                c.snapshot()["correction"]["status"], "unavailable"
            )
            c.start({"mode": "score"})
            self.assertEqual(set(c.workers), {"score"})
        finally:
            c.stop()

    def test_late_result_requires_current_generation_epoch_and_connection(self):
        c = self.make_coordinator()
        c.generation = 5
        c.state = {"running": True, "status": "running", "epoch": 3}
        now = time.monotonic()
        for generation, epoch in [(4, 3), (5, 2)]:
            c.receive(
                "score",
                {
                    "generation": generation,
                    "epoch": epoch,
                    "timestamp": now,
                    "result": "wrong",
                },
            )
            self.assertNotIn("score", c.state)
        result = {
            "generation": 5,
            "epoch": 3,
            "timestamp": now,
            "status": "collecting",
        }
        c.receive("score", result)
        self.assertEqual(c.state["score"], result)
        c.state["status"] = "connecting"
        c.receive("score", {**result, "timestamp": now + 1, "result": "wrong"})
        self.assertEqual(c.state["score"], result)
        c.state["score"]["timestamp"] = now - 2
        self.assertEqual(c.snapshot()["score"]["status"], "stale")

    def test_fixed_mode_and_invalid_options(self):
        self.assertEqual(validate({}, "score")["mode"], "score")
        for options in (
            {"mode": "correction"},
            {"mode": "combined"},
            {"exercise": "invalid"},
            {"camera": "other"},
            {"webcam": "-1"},
            {"esp32": "file:///etc/passwd"},
            {"view": "side"},
        ):
            with self.assertRaises(ValueError):
                validate(options, "score")

    def test_auto_fallback_uses_new_epoch_and_closes_esp32(self):
        readers = []

        class Reader:
            def __init__(self, source):
                self.source = source
                self.closed = False
                readers.append(self)

            def start(self):
                pass

            def stop(self):
                self.closed = True

            def snapshot(self):
                return 0, 1, None, False, "fixture", 0

        c = ChoiceCamera(
            validate({"camera": "auto"}), factory=Reader, timeout=0
        )
        c.start()
        try:
            result = c.snapshot()
            self.assertEqual(c.selected, "webcam")
            self.assertTrue(readers[0].closed)
            self.assertEqual(readers[1].source, "0")
            self.assertGreater(result[1], 1)
        finally:
            c.stop()

    def test_http_idle_does_not_start_camera_and_rejects_foreign_start(self):
        c = self.make_coordinator()
        server = make_server(0, c)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(base) as response:
                self.assertEqual(json.load(response)["mode"], "score")
            with urllib.request.urlopen(base + "/api/state") as response:
                self.assertFalse(json.load(response)["running"])
            self.assertIsNone(c.camera)
            for payload, origin, expected in [
                ({}, "https://example.com", 403),
                ([], base, 400),
                ({"exercise": "unknown"}, base, 400),
            ]:
                request = urllib.request.Request(
                    base + "/api/start",
                    json.dumps(payload).encode(),
                    {"Content-Type": "application/json", "Origin": origin},
                )
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(request)
                self.assertEqual(error.exception.code, expected)
                error.exception.close()
            request = urllib.request.Request(
                base + "/api/start",
                b'{"mode":"score"}',
                {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request) as response:
                self.assertEqual(
                    json.load(response)["options"]["mode"], "score"
                )
        finally:
            c.stop()
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
