"""미관측·구도 부족이 임의 총점으로 바뀌지 않는 경계를 검사한다."""

import pathlib
import hashlib
import json
import tempfile
import unittest
from unittest import mock

import numpy as np
from sklearn.dummy import DummyRegressor

import geometry
import rule_score
import audit
import benchmark


class AssessmentTest(unittest.TestCase):
    """실제 점수 신뢰도에 영향을 주는 판정 불가 조건을 검사한다."""

    def test_missing_joints_never_become_full_or_zero_score(self) -> None:
        """관절 전체 누락은 정상·오류 점수가 아닌 None이어야 한다."""
        result = rule_score.evaluate(
            "pushups",
            np.full((30, 33, 2), np.nan),
            None,
            ["trunk_line", "descent_range", "return_extension", "support"],
            [40, 30, 20, 10],
        )
        self.assertIsNone(result["score"])
        self.assertEqual(result["levels"], [None, None, None, None])

    def test_squat_side_only_cannot_generate_multiview_total(self) -> None:
        """측면만 있을 때 정면 판정 항목을 추측하지 않아야 한다."""
        result = rule_score.evaluate(
            "squats",
            np.ones((30, 33, 2)),
            None,
            ["descent_range", "return_extension", "knee_foot", "trunk_balance"],
            [30, 20, 30, 20],
        )
        self.assertIsNone(result["score"])
        self.assertEqual(result["levels"][2:], [None, None])

    def test_zero_length_limb_has_unknown_angle(self) -> None:
        """겹친 관절의 영벡터가 유효 각도 0도로 바뀌지 않아야 한다."""
        points = np.ones((5, 2))
        self.assertTrue(
            np.isnan(geometry.angle_deg(points, points, points)).all()
        )

    def test_weight_contract_mismatch_is_rejected(self) -> None:
        """잘못된 배점의 결과는 점수로 저장하지 않아야 한다."""
        with self.assertRaises(ValueError):
            rule_score.evaluate(
                "plank", np.ones((30, 33, 2)), None, ["trunk_line"], [40]
            )

    def test_whole_missing_contact_joint_holds_both_methods(self) -> None:
        """평균 관측률이 높아도 발끝 전체 누락을 두 방식 공통 보류한다."""
        sequence = np.ones((30, 33, 2))
        sequence[:, [31, 32]] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            folder = root / "heavy" / "example"
            folder.mkdir(parents=True)
            config = dict(
                schema=audit.CACHE_SCHEMA, source_sha256="source", segment={}
            )
            fingerprint = hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest()
            audit.save_json(
                folder / "side.json",
                dict(config=config, fingerprint=fingerprint),
            )
            np.savez(
                folder / "side.npz",
                fingerprint=fingerprint,
                image_xy=sequence,
                frame_numbers=np.arange(30),
            )
            with mock.patch.object(audit, "OUT", root):
                _, _, reasons = benchmark.load_input(
                    dict(
                        id="example",
                        exercise="pushups",
                        video_sha256="source",
                        segment={},
                    )
                )
        self.assertIn(
            "primary_required_joint_observation_below_80_percent", reasons
        )

    def test_obsolete_cache_stops_model_input(self) -> None:
        """이전 추출 버전의 캐시가 현재 모델 입력에 섞이지 않아야 한다."""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "side.npz"
            config = dict(schema="obsolete", source_sha256="source", segment={})
            fingerprint = hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest()
            audit.save_json(
                path.with_suffix(".json"),
                dict(config=config, fingerprint=fingerprint),
            )
            with self.assertRaises(ValueError):
                benchmark.validate_cache(
                    path, dict(video_sha256="source", segment={}), "side"
                )

    def test_participant_fold_does_not_see_held_labels(self) -> None:
        """제외 참여자의 등급이 그 폴드 평균 예측에 유입되지 않아야 한다."""
        rows = [
            dict(
                manifest=dict(
                    id=str(i),
                    person=str(i),
                    video_sha256=str(i),
                    front_sha256=None,
                ),
                x=np.array([i]),
                label=dict(
                    score=i * 50,
                    criterion_levels=[i] * 4,
                    criterion_weights=[25] * 4,
                ),
            )
            for i in range(3)
        ]
        result = benchmark.grouped_diagnostic(DummyRegressor(), rows)
        folds = {r["held_person"]: r for r in result["folds"]}
        self.assertEqual(folds["2"]["predictions"][0]["predicted_score"], 25)
        self.assertEqual(folds["0"]["predictions"][0]["predicted_score"], 75)
        self.assertFalse(result["independent_final_test"])
        rows[1]["manifest"]["video_sha256"] = "0"
        with self.assertRaises(ValueError):
            benchmark.grouped_diagnostic(DummyRegressor(), rows)
        rows[1]["manifest"]["video_sha256"] = "1"
        rows[1]["manifest"]["source_original_sha256"] = "0"
        with self.assertRaises(ValueError):
            benchmark.grouped_diagnostic(DummyRegressor(), rows)


if __name__ == "__main__":
    unittest.main()
