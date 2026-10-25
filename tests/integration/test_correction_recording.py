"""저장 입력 무결성·관측 누락·원본 교정 루프 보존을 검사합니다."""

import ast
import json
import tempfile
import unittest
from pathlib import Path

from GUI.FitHelper.correction_session_store import SessionReader, SessionWriter


class Array:
    def tolist(self):
        return [[0.1, 0.2, -0.3, 0.9] for _ in range(33)]


class RecordingTests(unittest.TestCase):
    def test_round_trip_preserves_missing_pose_and_current_prediction(self):
        with tempfile.TemporaryDirectory() as root:
            writer = SessionWriter(root, {"exercise": "squats"})
            prediction = {
                "scores": {"shallow": 0.8},
                "thresholds": {"shallow": 0.25},
                "detected_errors": ["shallow"],
                "feedbacks": ["Depth"],
            }
            base = dict(
                status="predicting",
                pose_visible=True,
                buffer_frames=32,
                required_frames=32,
                result=prediction,
            )
            writer.append(Array(), None, 10.0, 1, 0, 640, 480, base)
            base.update(
                status="pose_not_visible", pose_visible=False, result=None
            )
            writer.append(None, None, 10.2, 2, 0, 640, 480, base)
            writer.finish({}, "stop_requested")
            result = SessionReader(root).read(
                writer.session_id, limit=200, verify=True
            )
            self.assertTrue(result["integrity_matches"])
            self.assertEqual(result["frames"][0]["landmarks"], Array().tolist())
            self.assertEqual(result["frames"][0]["prediction"], prediction)
            self.assertIsNone(result["frames"][1]["landmarks"])
            self.assertIsNone(result["frames"][1]["prediction"])
            self.assertEqual(result["summary"]["prediction_count"], 1)
            self.assertEqual(result["summary"]["missing_pose_frames"], 1)
            self.assertIsNone(result["summary"]["actual_repetition_count"])
            self.assertEqual(
                SessionReader(root).read(writer.session_id, offset=1)["frames"][
                    0
                ]["index"],
                1,
            )

    def test_tampering_is_detected_after_reopening_store(self):
        with tempfile.TemporaryDirectory() as root:
            writer = SessionWriter(root, {"exercise": "squats"})
            writer.finish({}, "stop_requested")
            with (writer.directory / "frames.jsonl").open("a") as stream:
                stream.write("{}\n")
            self.assertFalse(
                SessionReader(root).read(writer.session_id, verify=True)[
                    "integrity_matches"
                ]
            )

    def test_crash_without_summary_is_never_claimed_completed(self):
        with tempfile.TemporaryDirectory() as root:
            writer = SessionWriter(root, {"exercise": "squats"})
            writer.frames.close()
            self.assertEqual(
                SessionReader(root).list_sessions()[0]["status"], "unfinalized"
            )
            self.assertIsNone(
                SessionReader(root).read(writer.session_id, verify=True)[
                    "integrity_matches"
                ]
            )

    def test_session_id_cannot_escape_root(self):
        with self.assertRaises(ValueError):
            SessionReader("/tmp").read("../secret")

    def test_runtime_error_is_recorded_as_failed(self):
        with tempfile.TemporaryDirectory() as root:
            writer = SessionWriter(root, {"exercise": "squats"})
            writer.finish({"error": "camera failed"}, "engine_exit")
            self.assertEqual(
                SessionReader(root).read(writer.session_id)["summary"][
                    "status"
                ],
                "failed",
            )

    def test_copied_inference_loop_only_adds_recording_hooks(self):
        repo = Path(__file__).resolve().parents[2]
        original = ast.parse(
            (
                repo
                / "Server/PostureAnalysis/posture-error-training/live_app/engine.py"
            ).read_text()
        )
        copied = ast.parse(
            (repo / "GUI/FitHelper/direct_correction_server.py").read_text()
        )

        def get_run(tree, name):
            return next(
                f
                for c in tree.body
                if isinstance(c, ast.ClassDef) and c.name == name
                for f in c.body
                if isinstance(f, ast.FunctionDef) and f.name == "run"
            )

        class Strip(ast.NodeTransformer):
            def visit_Expr(self, node):
                if (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr
                    in ("begin_recording", "capture_frame", "finish_recording")
                ):
                    return None
                return node

        self.assertEqual(
            ast.dump(get_run(original, "LiveEngine")),
            ast.dump(Strip().visit(get_run(copied, "RecordingMixin"))),
        )


if __name__ == "__main__":
    unittest.main()
