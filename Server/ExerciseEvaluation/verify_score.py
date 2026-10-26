"""보존된 스쿼트 관절을 재생해 반복 수·항목 점수·총점 일치를 검사합니다."""

import json
from unittest.mock import patch

import numpy as np

from .paths import SCORE_LIVE
from .score_adapter import PoseScoreAdapter


def verify():
    """실제 저장값과 비교하며 재생 일치를 운동 정확도로 해석하지 않습니다."""
    folder = SCORE_LIVE / "sessions/1792858741575615610"
    state = json.loads((folder / "final-state.json").read_text())
    adapter = PoseScoreAdapter("squats")
    with np.load(folder / "pose.npz") as data:
        for tick, xy in zip(data["frame_numbers"], data["image_xy"]):
            adapter.process_points(
                xy, int(tick), int(data["width"]), int(data["height"])
            )
    reps = adapter.flush()
    assert adapter.tracker.count == state["movement"]["count"] == 7
    assert len(reps) == len(state["repetitions"]) == 7
    for actual, saved in zip(reps, state["repetitions"]):
        assert actual["number"] == saved["number"]
        assert actual["score_100"]["ai"] == saved["score_100"]["ai"]
        assert actual["score_100"]["rule"] == saved["score_100"]["rule"]
        for a, b in zip(
            actual["score_100"]["items"], saved["score_100"]["items"]
        ):
            for key in ("id", "weight", "ai_points", "rule_points"):
                assert a[key] == b[key], (actual["number"], key, a[key], b[key])
    # 과거 창이 남아도 현재 관절이 누락되면 유지 운동의 잠정 점수를 숨깁니다.
    hold = PoseScoreAdapter("plank")
    hold.available = [(i, np.zeros((33, 2))) for i in range(10)]
    hold.last_tick = 9
    hold.live_score = {"previous_window": True}
    with patch(
        "Server.ExerciseEvaluation.score_adapter.score_samples"
    ) as calculate:
        snapshot = hold.process_points(np.full((33, 2), np.nan), 10, 640, 480)
        assert snapshot["live_score_100"] is None
        calculate.assert_not_called()
    print(
        json.dumps(
            {
                "session": folder.name,
                "samples": 690,
                "count": 7,
                "all_item_points_equal": True,
                "all_ai_and_rule_totals_equal": True,
                "missing_current_pose_hides_provisional_hold_score": True,
                "ai_totals": [r["score_100"]["ai"]["total"] for r in reps],
                "scope": "저장 관절 재생 회귀 검사입니다. 새 촬영의 정확도 검증이 아닙니다.",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    verify()
