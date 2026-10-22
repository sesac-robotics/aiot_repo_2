"""QEVD 제공자 문자열 보존과 점수 추론의 분리를 검증한다."""

import contextlib
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from app.scoring import media
from app.scoring.__main__ import demo_sequence, main


class CliTest(unittest.TestCase):
    """카메라·네트워크 없이 실제 CLI 입출력을 검사한다."""

    def test_provider_labels_do_not_set_scores(self) -> None:
        """good·bad 문자열을 바꿔도 같은 관절의 점수는 바뀌지 않는다."""
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            landmarks = root / "pose.npz"
            media.save_landmarks(landmarks, demo_sequence())
            scores = []
            for label in ("good", "bad"):
                request = dict(
                    video_path="./not-distributed.mp4",
                    labels=[label],
                    labels_descriptive=[label + " provider description"],
                    split="validation",
                    assessment_input=dict(
                        exercise="shoulder", view="front", complete_attempt=True
                    ),
                )
                path = root / (label + ".json")
                output = root / label
                path.write_text(json.dumps(request), encoding="utf-8")
                with (
                    mock.patch(
                        "sys.argv",
                        [
                            "scoring",
                            "--request",
                            str(path),
                            "--landmarks",
                            str(landmarks),
                            "--output",
                            str(output),
                        ],
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(main(), 0)
                result = json.loads((output / "result.json").read_text())
                self.assertTrue(result["output_complete"])
                self.assertEqual(
                    result["qevd"],
                    {
                        k: request[k]
                        for k in (
                            "video_path",
                            "labels",
                            "labels_descriptive",
                            "split",
                        )
                    },
                )
                scores.append((result["ai_score"], result["rule_score"]))
            self.assertEqual(scores[0], scores[1])
            self.assertIsNotNone(scores[0][0])

    def test_invalid_request_has_no_completed_output(self) -> None:
        """자료 계약이 잘못되면 성공 폴더나 완료 JSON을 만들지 않는다."""
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            request = root / "invalid.json"
            request.write_text("{}", encoding="utf-8")
            output = root / "output"
            with (
                mock.patch(
                    "sys.argv",
                    [
                        "scoring",
                        "--request",
                        str(request),
                        "--output",
                        str(output),
                    ],
                ),
                self.assertLogs(level="ERROR"),
            ):
                self.assertEqual(main(), 2)
            self.assertFalse(output.exists())
