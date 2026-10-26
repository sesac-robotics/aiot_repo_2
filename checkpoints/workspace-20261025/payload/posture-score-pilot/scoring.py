"""측면 영상의 2차원 관절 측정값으로 실습용 반복 점수를 계산한다.

각도 단위는 도, 시간은 초이다. 누락 관절은 보간하지 않으며, 반복이
완료되지 않으면 점수를 만들지 않는다. 배점은 전문가 미검증 초안이다.
"""

import math
import numpy as np


def angle(a, b, c):
    """세 점의 중간 점 내각을 도 단위로 반환한다.

    Args:
        a, b, c: 같은 좌표계의 2차원 점.
    Returns:
        0~180도의 내각. 선분 길이가 0이면 NaN.
    """
    u, v = np.asarray(a) - b, np.asarray(c) - b
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return (
        float(np.degrees(np.arccos(np.clip(u @ v / d, -1, 1))))
        if d > 1e-6
        else math.nan
    )


def measure(points, side, exercise):
    """보이는 한쪽 관절에서 운동별 각도를 측정한다.

    Args:
        points: 픽셀 좌표를 담은 관절 번호별 사전.
        side: left 또는 right. 영상 내에서 고정한다.
        exercise: squat 또는 pushup.
    Returns:
        각도 사전 또는 None. 필요한 관절 누락ㆍ자세 방향 불일치 시 None.
    """
    s, e, w, h, k, a = (
        (11, 13, 15, 23, 25, 27) if side == "left" else (12, 14, 16, 24, 26, 28)
    )
    required = [s, h, k, a] if exercise == "squat" else [s, e, w, h, k, a]
    if not all(i in points for i in required):
        return None
    shoulder, hip, ankle = points[s], points[h], points[a]
    axis = np.asarray(shoulder) - ankle
    if np.linalg.norm(axis) < 30:
        return None
    if exercise == "squat" and abs(axis[1]) < abs(axis[0]):
        return None
    if exercise == "pushup" and abs(axis[0]) < abs(axis[1]):
        return None
    primary = (
        angle(points[h], points[k], points[a])
        if exercise == "squat"
        else angle(points[s], points[e], points[w])
    )
    torso = np.asarray(shoulder) - hip
    inclination = float(np.degrees(np.arctan2(abs(torso[0]), abs(torso[1]))))
    alignment = angle(shoulder, hip, ankle)
    values = {"primary": primary, "torso": inclination, "alignment": alignment}
    return values if all(math.isfinite(v) for v in values.values()) else None


def ramp(value, bad, good):
    """측정값을 단위 없는 0~1 값으로 변환한다.

    Args:
        value: 각도 또는 변동량. bad, good과 같은 단위.
        bad: 0점 경계. good: 만점 경계.
    Returns:
        경계 사이 선형 보간값. 범위 밖은 0 또는 1로 제한한다.
    """
    return float(np.clip((value - bad) / (good - bad), 0, 1))


class RepetitionTracker:
    """기립ㆍ팔 폄 → 굽힘 → 폄의 완결 구간만 기록한다.

    누락 구간이 0.25초를 넘으면 진행 중 구간을 폐기한다.
    """

    def __init__(self, exercise, rules):
        """운동 이름과 초ㆍ도 단위 기준값으로 추적 상태를 초기화한다."""
        self.exercise, self.rules = exercise, rules
        self.ready = False
        self.active = False
        self.segment = []
        self.last_valid = None
        self.hold = None
        self.last_top = None
        self.results = []
        self.discarded = 0

    def update(self, t, values):
        """한 시점의 측정값으로 반복 상태와 점수를 갱신한다.

        Args:
            t: 영상 시작부터의 초.
            values: measure 결과 또는 측정 불가를 뜻하는 None.
        Returns:
            새 반복 결과 사전 또는 None. 결과는 results에도 누적한다.
        """
        if values is None:
            self.hold = None
            if (
                self.last_valid is not None
                and t - self.last_valid > self.rules["maximum_missing_seconds"]
            ):
                self.discarded += int(self.active)
                self.ready = self.active = False
                self.segment = []
                self.hold = None
            return None
        self.last_valid = t
        p = values["primary"]
        top = p >= self.rules["top_angle"]
        if top:
            if self.hold is None:
                self.hold = t
        else:
            self.hold = None
        if not self.active:
            if top and t - self.hold >= 0.2:
                self.ready = True
                self.last_top = (t, values)
            if self.ready and p < self.rules["start_angle"]:
                self.active = True
                self.segment = [self.last_top, (t, values)]
            return None
        self.segment.append((t, values))
        if t - self.segment[0][0] > 20:
            self.discarded += 1
            self.active = self.ready = False
            self.segment = []
            return None
        if top and t - self.hold >= 0.2:
            data = self.segment
            self.active = False
            self.ready = True
            self.last_top = (t, values)
            duration = t - data[0][0]
            angles = np.array([v["primary"] for _, v in data])
            if duration < 0.6 or np.ptp(angles) < self.rules["minimum_range"]:
                self.discarded += 1
                return None
            cfg = self.rules[self.exercise]
            depth = float(np.percentile(angles, 10))
            extension = float(np.percentile(angles, 90))
            depth_score = ramp(depth, cfg["depth_zero"], cfg["depth_full"])
            extension_score = ramp(
                extension, cfg["extension_zero"], cfg["extension_full"]
            )
            if self.exercise == "squat":
                body = float(np.percentile([v["torso"] for _, v in data], 90))
                body_score = ramp(body, cfg["torso_zero"], cfg["torso_full"])
                body_name = "몸통 기울기"
            else:
                body = float(
                    np.percentile([v["alignment"] for _, v in data], 10)
                )
                body_score = ramp(
                    body, cfg["alignment_zero"], cfg["alignment_full"]
                )
                body_name = "몸통 정렬"
            # 같은 0.1초 간격으로 재표본화해 영상 FPS 차이 영향을 줄인다.
            times = np.array([x[0] for x in data])
            grid = np.arange(times[0], times[-1], 0.1)
            sampled = np.interp(grid, times, angles)
            jitter = float(np.percentile(np.abs(np.diff(sampled, n=2)), 90))
            smooth = ramp(jitter, 15, 3)
            names = [
                "내려가는 깊이",
                "다시 펴기",
                body_name,
                "각도 변화 안정성",
            ]
            parts = [
                {"name": n, "score": round(s * w, 1), "maximum": w}
                for n, s, w in zip(
                    names,
                    [depth_score, extension_score, body_score, smooth],
                    cfg["weights"],
                )
            ]
            result = {
                "number": len(self.results) + 1,
                "start": round(data[0][0], 2),
                "end": round(t, 2),
                "score": round(sum(p["score"] for p in parts), 1),
                "parts": parts,
                "measurements": {
                    "minimum_angle_p10": round(depth, 1),
                    "extension_p90": round(extension, 1),
                    "body_angle": round(body, 1),
                    "jitter": round(jitter, 2),
                },
            }
            self.results.append(result)
            return result
        return None
