"""재학습에 최종 시험과 Rule 출력 정답이 섞이지 않는지 검사한다."""

import copy
import unittest

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from app.scoring.training import refit_heads


def records() -> list[dict]:
    """네 독립 그룹의 합성 학습 입력을 만든다. 정확도 검증 자료가 아니다."""
    rows = []
    for i in range(8):
        rows.append(
            dict(
                id=f"r{i}",
                exercise="ytw",
                person=f"p{i // 2}",
                sources=[format(i + 1, "064x")],
                split="train",
                label_origin="direct_visual_draft",
                levels=[2 if i >= 4 else 0, None, None, None],
                items=[
                    dict(
                        item="y_reach",
                        bank="front",
                        eligible=True,
                        feature=[float(i)],
                    )
                ]
                + [
                    dict(
                        item="unused",
                        bank="front",
                        eligible=False,
                        feature=None,
                    )
                    for _ in range(3)
                ],
            )
        )
    return rows


class TrainingTest(unittest.TestCase):
    """분리된 자료만 학습하고 기존 Rule을 보존하는지 검사한다."""

    def test_final_test_and_rule_targets_rejected(self) -> None:
        """최종 시험이나 Rule 출력 출처를 주면 학습 전에 실패한다."""
        for change in (
            {"split": "final_test"},
            {"label_origin": "rule_output"},
        ):
            rows = records()
            rows[0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                refit_heads({}, rows)

    def test_validation_targets_do_not_select_model(self) -> None:
        """개발 검증 정답을 바꿔도 학습된 후보와 예측은 변하지 않는다."""
        rows = records()
        validation = copy.deepcopy(rows[0])
        validation.update(
            id="validation",
            person="other",
            sources=["f" * 64],
            split="validation",
        )
        # 검증 특징을 학습에 넣으면 변환 단계에서 실패하도록 만든다.
        validation["items"][0]["feature"] = ["must-not-fit"]
        rows.append(validation)
        template = {0: dict(model=ExtraTreesRegressor(), policy=[1.0, 2.0])}
        first, _ = refit_heads(template, rows)
        rows[-1]["levels"][0] = 1
        second, report = refit_heads(template, rows)
        self.assertTrue(
            np.array_equal(
                first[0]["model"].predict([[1.0], [6.0]]),
                second[0]["model"].predict([[1.0], [6.0]]),
            )
        )
        self.assertEqual(first[0]["policy"], template[0]["policy"])
        self.assertFalse(report["validation_used_for_fitting"])
        self.assertFalse(report["final_test_used"])
