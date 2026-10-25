"""학습된 운동별 준비 자세와 시간 순서의 관측 근거를 계산합니다.

Learn the exercise's ready pose; temporal return is an ordered event.
A frame away from ready is not a failed repetition. No angle rule supplies a grade.
"""

from dataclasses import dataclass

import numpy as np

JOINTS = {
    "pushups": [11, 12, 13, 14, 15, 16, 23, 24],
    "shoulder": [11, 12, 13, 14, 15, 16, 23, 24],
    "ytw": [11, 12, 13, 14, 15, 16, 23, 24],
    "squats": [11, 12, 23, 24, 25, 26, 27, 28],
    "plank": [11, 12, 13, 14, 23, 24, 25, 26, 27, 28],
}
READY_MEANING = {
    "pushups": "arm_extension_ready_other_items_scored_separately",
    "shoulder": "shoulder_level_rack_pose",
    "ytw": "protocol_defined_Y_start_phase_not_forced_arms_down",
    "squats": "standing_pose",
    "plank": "plank_support_hold_not_standing_return",
}


def pose(exercise, xy):
    a = np.asarray(xy, float)
    if a.shape != (33, 2):
        raise ValueError("Expected33x2 image coordinates")
    if exercise in ["pushups", "plank", "squats"]:
        # Side views need the actually visible chain; occluded far limbs are never filled.
        required = (
            [11, 13, 15, 23]
            if exercise == "pushups"
            else (
                [11, 23, 25, 27]
                if exercise == "squats"
                else [11, 13, 23, 25, 27]
            )
        )
        options = [
            offset
            for offset in [0, 1]
            if np.isfinite(a[[j + offset for j in required]]).all()
        ]
        if not options:
            return None
        offset = options[0]
        joints = [j + offset for j in required]
        hip = a[23 + offset]
        shoulder = a[11 + offset]
        triples = (
            [(11 + offset, 13 + offset, 15 + offset)]
            if exercise == "pushups"
            else (
                [
                    (11 + offset, 23 + offset, 25 + offset),
                    (23 + offset, 25 + offset, 27 + offset),
                ]
                if exercise == "squats"
                else [
                    (11 + offset, 23 + offset, 27 + offset),
                    (23 + offset, 25 + offset, 27 + offset),
                    (13 + offset, 11 + offset, 23 + offset),
                ]
            )
        )
    else:
        joints = JOINTS[exercise]
        if not np.isfinite(a[joints]).all():
            return None
        hip = a[[23, 24]].mean(0)
        shoulder = a[[11, 12]].mean(0)
        triples = (
            [(11, 13, 15), (12, 14, 16)]
            if exercise in ["shoulder", "ytw"]
            else [(11, 23, 25), (12, 24, 26), (23, 25, 27), (24, 26, 28)]
        )
    scale = np.linalg.norm(shoulder - hip)
    if scale < 1:
        return None
    normal = (a[joints] - hip) / scale
    axis = (shoulder - hip) / scale
    # Align horizontal mirroring using torso direction, never the better angle/score.
    if axis[0] < 0:
        normal[:, 0] *= -1
        axis[0] *= -1
    angles = []
    for h, c, w in triples:
        u = a[h] - a[c]
        v = a[w] - a[c]
        den = np.linalg.norm(u) * np.linalg.norm(v)
        if den < 1:
            return None
        angles.append(
            float(np.arccos(np.clip(np.dot(u, v) / den, -1, 1)) / np.pi)
        )
    anchor = (
        [hip[1] / scale, shoulder[1] / scale] if exercise == "squats" else []
    )
    # Preserve the height anchor for the learned comparison to the initial frame.
    return np.r_[normal.flatten(), angles, axis, anchor]


def relative(x, reference):
    x = np.asarray(x, float)
    r = np.asarray(reference, float)
    if (
        x.shape != r.shape
        or x.ndim != 1
        or not np.isfinite(x).all()
        or not np.isfinite(r).all()
    ):
        raise ValueError("Observed aligned pose vectors required")
    return np.r_[x, x - r, abs(x - r)]


@dataclass
class ReturnTracker:
    exercise: str
    high: float = 0.8
    low: float = 0.2
    dwell: int = 3
    max_gap_seconds: float = 0.35

    def __post_init__(self):
        if self.exercise not in JOINTS:
            raise ValueError("Unknown exercise")
        self.ready = False
        self.active = False
        self.uncertain = False
        self.hi = 0
        self.lo = 0
        self.return_events = 0
        self.last_time = None
        self.last_observed = None
        self.status = "waiting_initial_ready"
        self.calibration_time = None

    def update(self, t, p_ready):
        if self.last_time is not None and t <= self.last_time:
            raise ValueError("Strictly increasing timestamps required")
        gap = (
            self.last_time is not None
            and t - self.last_time > self.max_gap_seconds
        )
        self.last_time = t
        if p_ready is None or not np.isfinite(p_ready) or gap:
            self.hi = self.lo = 0
            if self.active:
                self.uncertain = True
            self.status = "unobserved"
            return self.snapshot()
        if not 0 <= p_ready <= 1:
            raise ValueError("Probability must be0..1")
        self.last_observed = t
        self.hi = self.hi + 1 if p_ready >= self.high else 0
        self.lo = self.lo + 1 if p_ready <= self.low else 0
        if not self.ready:
            if self.hi >= self.dwell:
                self.ready = True
                self.calibration_time = t
                self.status = (
                    "holding" if self.exercise == "plank" else "initial_ready"
                )
        elif self.exercise == "plank":
            if self.lo >= self.dwell:
                self.status = "support_change_observed"
            elif self.hi >= self.dwell:
                self.status = "holding"
        elif not self.active and self.lo >= self.dwell:
            self.active = True
            self.uncertain = False
            self.status = "movement_away_from_initial"
        elif self.active and self.hi >= self.dwell:
            self.status = (
                "return_observed_cycle_unverified"
                if self.uncertain
                else "returned_to_initial"
            )
            if not self.uncertain:
                self.return_events += 1
            self.active = False
            self.uncertain = False
            self.lo = 0
        return self.snapshot()

    def finish(self, *, attempt_closed=False, user_stopped=False):
        # Stream truncation / user stop never manufactures a failure label or grade0.
        if user_stopped:
            self.status = "stopped_by_user"
        elif self.exercise == "plank":
            self.status = "hold_observation_ended"
        elif attempt_closed and self.active:
            self.status = (
                "return_unverified"
                if self.uncertain
                else "closed_attempt_return_not_observed"
            )
        else:
            self.status = "observation_ended"
        return self.snapshot()

    def snapshot(self):
        return dict(
            status=self.status,
            return_events=self.return_events,
            valid_repetition=None,
            initial_ready_observed=self.ready,
            initial_ready_time=self.calibration_time,
            grade=None,
            grade_reason="state_and_event_evidence_not_a_trained_severity_grade",
            exercise=self.exercise,
        )


class YTWPhaseTracker:
    """Count observed Y->T->W ordering; do not require invented arms-down rests."""

    def __init__(self, dwell=3, confidence=0.8):
        self.dwell = dwell
        self.confidence = confidence
        self.candidate = None
        self.streak = 0
        self.expected = "Y"
        self.cycles = 0
        self.last_time = None
        self.status = "waiting_Y"

    def update(self, t, probabilities):
        if self.last_time is not None and t <= self.last_time:
            raise ValueError("Increasing times required")
        gap = self.last_time is not None and t - self.last_time > 0.35
        self.last_time = t
        if probabilities is None or gap:
            self.candidate = None
            self.streak = 0
            self.expected = "Y"
            self.status = "phase_unobserved"
            return self.snapshot()
        if set(probabilities) != {"Y", "T", "W"} or any(
            not np.isfinite(v) or not 0 <= v <= 1
            for v in probabilities.values()
        ):
            raise ValueError("Observed Y/T/W probabilities required")
        phase = max(probabilities, key=probabilities.get)
        if probabilities[phase] < self.confidence:
            self.candidate = None
            self.streak = 0
            return self.snapshot()
        self.streak = self.streak + 1 if phase == self.candidate else 1
        self.candidate = phase
        if self.streak != self.dwell:
            return self.snapshot()
        if phase == "Y":
            self.expected = "T"
            self.status = "Y_observed"
        elif phase == "T" and self.expected == "T":
            self.expected = "W"
            self.status = "Y_then_T_observed"
        elif phase == "W" and self.expected == "W":
            self.cycles += 1
            self.expected = "Y"
            self.status = "Y_T_W_order_completed"
        return self.snapshot()

    def snapshot(self):
        return dict(
            status=self.status,
            ordered_cycles=self.cycles,
            grade=None,
            phase_order_is_not_full_quality_grade=True,
        )


class InitialPoseAnalyzer:
    """Inference from a trusted, already loaded local state bundle; never fit online."""

    def __init__(self, bundle):
        if bundle.get("schema") != "reviewed_initial_ready_and_return_state_v3":
            raise ValueError("Unsupported learned-ready bundle")
        self.heads = bundle["heads"]

    def run(
        self,
        exercise,
        xy,
        times,
        *,
        view,
        protocol,
        attempt_closed=False,
        user_stopped=False,
    ):
        expected = {
            "pushups": ("side", "arm_return_v1"),
            "shoulder": ("front", "seated_rack_v1"),
            "ytw": ("front", "upright_Y_T_W_v1"),
            "plank": ("side", "support_hold_v1"),
            "squats": ("front", "standing_return_v1"),
        }
        if exercise not in expected:
            raise ValueError("Unknown exercise")
        if (view, protocol) != expected[exercise]:
            return dict(
                status="unassessed_protocol_or_view_not_trained",
                grade=None,
                production_replacement=False,
            )
        frames = np.asarray(xy, float)
        ts = np.asarray(times, float)
        if (
            frames.ndim != 3
            or frames.shape[1:] != (33, 2)
            or ts.shape != (len(frames),)
            or not np.isfinite(ts).all()
            or np.any(np.diff(ts) <= 0)
        ):
            raise ValueError("Observed sequence and increasing times required")
        absolute = self.heads.get((exercise, "x", "target"))
        if absolute is None:
            return dict(
                status="unassessed_state_training_support_missing",
                grade=None,
                production_replacement=False,
            )
        if absolute.get("state_gate_passed") is not True:
            return dict(
                status="unassessed_state_development_gate_failed",
                grade=None,
                production_replacement=False,
            )
        tracker = ReturnTracker(exercise)
        phase_tracker = YTWPhaseTracker() if exercise == "ytw" else None
        reference = None
        recent = []
        evidence = []
        for t, a in zip(ts, frames):
            x = pose(exercise, a)
            prob = None
            decision = None
            phase_prob = None
            phase_evidence = None
            bank = "x"
            if x is not None:
                h = absolute
                vector = x
                relative_head = self.heads.get(
                    (exercise, "relative_x", "target")
                )
                if (
                    reference is not None
                    and relative_head is not None
                    and relative_head.get("state_gate_passed") is True
                ):
                    if reference["time"] >= t:
                        raise ValueError(
                            "Reference must precede assessed frame"
                        )
                    bank = "relative_x"
                    h = relative_head
                    vector = relative(x, reference["x"])
                classes = list(h["model"].classes_)
                prob = float(
                    h["model"].predict_proba([vector])[0, classes.index(1)]
                )
                decision = int(h["model"].predict([vector])[0])
                if reference is None:
                    recent = (
                        (recent + [(float(t), x)])[-3:] if decision == 1 else []
                    )
                    if (
                        len(recent) == 3
                        and recent[-1][0] - recent[0][0] <= 0.25
                    ):
                        reference = dict(
                            time=float(t),
                            x=np.mean([v for _, v in recent], axis=0),
                        )
                if phase_tracker is not None:
                    # The phase classifier is selected from absolute training-group CV.
                    phase_head = self.heads.get(("ytw", "x", "phase"))
                    if (
                        phase_head is not None
                        and phase_head.get("state_gate_passed") is True
                    ):
                        phase_prob = {
                            str(c): float(p)
                            for c, p in zip(
                                phase_head["model"].classes_,
                                phase_head["model"].predict_proba([x])[0],
                            )
                        }
                        predicted_phase = str(
                            phase_head["model"].predict([x])[0]
                        )
                        phase_evidence = {
                            c: float(c == predicted_phase)
                            for c in ["Y", "T", "W"]
                        }
            else:
                recent = []
            # Use the learned class decision plus temporal dwell; uncalibrated soft
            # probabilities are reported, not mistaken for a universal confidence cutoff.
            state = tracker.update(
                float(t), None if decision is None else float(decision)
            )
            phase = (
                phase_tracker.update(float(t), phase_evidence)
                if phase_tracker is not None
                else None
            )
            evidence.append(
                dict(
                    time=float(t),
                    ready_probability=prob,
                    predicted_ready_state=decision,
                    phase_probabilities=phase_prob,
                    feature_bank=bank,
                    status=state["status"],
                    phase=phase,
                )
            )
        final = tracker.finish(
            attempt_closed=attempt_closed, user_stopped=user_stopped
        )
        return dict(
            schema="learned_initial_pose_return_evidence_v1",
            state=final,
            phase_order=(
                phase_tracker.snapshot() if phase_tracker is not None else None
            ),
            calibration_time=reference["time"] if reference else None,
            frames=evidence,
            grade=None,
            scope="state_and_order_evidence_not_trained_0_1_2_return_grade",
            production_replacement=False,
        )
