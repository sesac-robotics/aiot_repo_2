"""이 구성요소의 입력·출력 및 실패 처리 계약을 검증합니다. 합성 입력이나 저장된 시험 자료의 검사 범위는 각 테스트 이름과 단언문에 명시합니다.

Five-exercise localhost integration test using actual cached held-out poses."""

import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from api_server import make_server
from suite_config import SUITE, TASKS


class FiveAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server(0)
        cls.thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_ready_and_four_actual_exercise_predictions(self):
        with urlopen(self.url + "/health") as f:
            self.assertTrue(all(json.load(f)["models_ready"].values()))
        for exercise in TASKS:
            with self.subTest(exercise=exercise):
                s = json.loads(
                    (SUITE / exercise / "outputs/splits.json").read_text()
                )
                vid = s["test"]["video_ids"][0]
                with np.load(SUITE / "cache" / f"{vid}.npz") as cache:
                    body = dict(
                        exercise=exercise,
                        landmarks=cache["landmarks"].tolist(),
                        aspect=float(cache["aspect"]),
                    )
                with urlopen(
                    Request(
                        self.url + "/predict",
                        data=json.dumps(body).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                ) as f:
                    r = json.load(f)
                    self.assertEqual(r["status"], "ok")
                    self.assertEqual(r["exercise"], exercise)
                    self.assertEqual(
                        len(r["scores"]), len(TASKS[exercise]["targets"])
                    )

    def test_invalid_exercise_rejected(self):
        body = dict(exercise="unknown", landmarks=np.ones((8, 33, 4)).tolist())
        with self.assertRaises(HTTPError) as e:
            urlopen(
                Request(self.url + "/predict", data=json.dumps(body).encode())
            )
        self.assertEqual(e.exception.code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
