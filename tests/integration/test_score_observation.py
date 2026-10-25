"""기준을 낮추지 않는 양 무릎 준비·독립 검출 채택·원시 기록 검증입니다."""

import json

import numpy as np
import pytest

from Server.ExerciseEvaluation.score_observation import (
    ScoringReadyGate,
    choose_recovery,
    inspect_pose,
)
from Server.ExerciseEvaluation.score_recording import (
    ScoreRecording,
    ScoreRecordingReader,
)


def pose():
    raw = np.full((33, 5), 0.7)
    for j, xy in {
        11: (0.4, 0.2),
        12: (0.6, 0.2),
        23: (0.43, 0.5),
        24: (0.57, 0.5),
        25: (0.43, 0.75),
        26: (0.57, 0.75),
    }.items():
        raw[j, :2] = xy
    return raw


def test_both_knees_required_without_lowering_confidence():
    raw = pose()
    raw[25, 3] = 0.49
    xy, q = inspect_pose(raw, 640, 480)
    assert not q["all_items_observed"] and not q["item_observed"]["knee_inward"]
    assert q["item_observed"]["trunk_balance"]
    assert np.isnan(xy[25]).all()
    assert q["joints"]["25"]["reasons"] == ["low_visibility"]


def test_border_and_presence_failures_are_distinguished():
    raw = pose()
    raw[25, 1] = 0.99
    raw[26, 4] = 0.49
    xy, q = inspect_pose(raw, 640, 480)
    assert "frame_border" in q["joints"]["25"]["reasons"]
    assert "low_presence" in q["joints"]["26"]["reasons"]
    assert not q["all_items_observed"]


def test_recovery_selects_one_whole_observed_pose_not_mixed_joints():
    primary = pose()
    primary[25, 3] = 0.1
    retry = pose()
    retry[0, 0] = 0.9
    selected, source, xy, q = choose_recovery(primary, retry, 640, 480)
    assert source == "independent_image_recovery" and q["all_items_observed"]
    assert np.array_equal(selected, retry)
    assert primary[25, 3] == 0.1


def test_wrong_person_recovery_is_rejected():
    primary = pose()
    primary[25, 3] = 0.1
    retry = pose()
    retry[:, 0] += 0.3
    selected, source, xy, q = choose_recovery(primary, retry, 640, 480)
    assert source == "video" and not q["all_items_observed"]


def test_valid_primary_is_never_replaced_by_retry():
    primary = pose()
    retry = pose()
    retry[0, 0] = 0.9
    selected, source, xy, q = choose_recovery(primary, retry, 640, 480)
    assert source == "video" and np.array_equal(selected, primary)


def test_ready_needs_six_real_consecutive_bilateral_samples():
    raw = pose()
    _, good = inspect_pose(raw, 640, 480)
    raw[25, 3] = 0.1
    _, bad = inspect_pose(raw, 640, 480)
    gate = ScoringReadyGate()
    for tick in range(5):
        assert not gate.step(tick, good)["prepared"]
    assert not gate.step(5, bad)["prepared"]
    for tick in range(6, 11):
        assert not gate.step(tick, good)["prepared"]
    assert gate.step(11, good)["prepared"]
    gate = ScoringReadyGate()
    for tick in [0, 2, 4, 6, 8, 10]:
        assert not gate.step(tick, good)["prepared"]


def test_recording_reopens_raw_confidence_rejection_and_missing_slots(tmp_path):
    raw = pose()
    raw[25, 3] = 0.1
    xy, q = inspect_pose(raw, 640, 480)
    writer = ScoreRecording(
        {"session_id": "a" * 32, "epoch": 1}, "squats", root=tmp_path
    )
    for tick in [0, 2]:
        writer.append(
            tick,
            tick / 10,
            {"shape": [480, 640, 3], "frame_id": tick},
            raw,
            None,
            raw,
            xy,
            q,
            "video",
            False,
            {},
        )
    writer.finish(
        {"assessment_summary": {"all_detected_repetitions_scored": False}},
        "test",
    )
    reader = ScoreRecordingReader(tmp_path)
    saved = reader.read(writer.metadata["recording_id"], 1, True)
    assert saved["integrity_matches"]
    assert saved["frame"]["raw_selected_landmarks"][25][3] == 0.1
    assert saved["frame"]["scoring_xy"][25] == [None, None]
    assert saved["frame"]["missing_slots_before"] == 1
    assert saved["frame"]["quality"]["joints"]["25"]["reasons"] == [
        "low_visibility"
    ]
    assert saved["summary"]["assessment_status"] == "not_complete"
    assert reader.list()[0]["recording_id"] == writer.metadata["recording_id"]
    with pytest.raises(ValueError):
        reader.read("../secret")


def test_live_ready_gate_releases_real_initial_frames_without_losing_first_motion():
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    import cv2

    from Server.ExerciseEvaluation.preview_privacy import encode_masked_preview

    tree = ast.parse(
        Path("Server/ExerciseEvaluation/score_adapter.py").read_text()
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ScoreAdapter"
    )
    function = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "process"
    )
    scope = dict(
        np=np,
        cv2=cv2,
        inspect_pose=inspect_pose,
        choose_recovery=choose_recovery,
        retry_due=__import__(
            "Server.ExerciseEvaluation.score_observation",
            fromlist=["retry_due"],
        ).retry_due,
        encode_masked_preview=encode_masked_preview,
        mp=SimpleNamespace(
            Image=lambda **kw: kw, ImageFormat=SimpleNamespace(SRGB=1)
        ),
    )
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            "live-process-test",
            "exec",
        ),
        scope,
    )
    seen = []

    def deliver(xy, tick, w, h):
        seen.append((tick, xy.copy()))
        return {}

    obj = SimpleNamespace(
        origin=None,
        extraction_last_tick=-1,
        exercise="squats",
        pose=SimpleNamespace(detect_for_video=lambda image, tick: pose()),
        raw_pose=lambda result: result,
        gate=ScoringReadyGate(),
        ready_buffer=[],
        had_reference=False,
        tracker=SimpleNamespace(standing_reference=None),
        last_preview=float("inf"),
        recording=None,
        snapshot=lambda: {},
        process_points=deliver,
    )
    frame = np.zeros((480, 640, 3), np.uint8)
    for tick in range(5):
        scope["process"](obj, frame, tick / 10)
    assert seen == []
    state = scope["process"](obj, frame, 0.5)
    assert [t for t, xy in seen] == list(range(6))
    assert state["readiness_released_actual_ticks"] == list(range(6))
    scope["process"](obj, frame, 0.6)
    assert [t for t, xy in seen] == list(range(7))
    assert all(
        np.allclose(xy[[25, 26]], [[275.2, 360], [364.8, 360]])
        for t, xy in seen
    )


def test_jittered_new_slot_still_redetects_a_present_person():
    from Server.ExerciseEvaluation.score_observation import retry_due

    raw = pose()
    raw[25, 3] = 0.4
    assert retry_due(raw, 46.571, 46.49)
    assert not retry_due(np.full((33, 5), np.nan), 46.571, 46.49)
    assert retry_due(np.full((33, 5), np.nan), 47.0, 46.49)
