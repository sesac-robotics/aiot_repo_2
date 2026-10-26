"""이미지 비율을 보정하고 유효한 실제 관절만 사용합니다."""

import math

from app.analysis.exercises.squat import knee_angle

SIDES = {"left": (11, 13, 15, 23, 25, 27), "right": (12, 14, 16, 24, 26, 28)}


class PoseGeometry:
    def __init__(self, keypoints, size, config):
        if (
            not isinstance(size, (tuple, list))
            or len(size) != 2
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                for v in size
            )
        ):
            raise ValueError("invalid_frame_size")
        self.size, self.config = size, config
        self.points, self.invalid = {}, {}
        for point in keypoints:
            if not isinstance(point, dict) or type(point.get("id")) is not int:
                continue
            ident = point["id"]
            if ident in self.points or ident in self.invalid:
                self.points.pop(ident, None)
                self.invalid[ident] = "duplicate_joint"
                continue
            reason = None
            if any(
                type(point.get(k)) not in (int, float)
                or not math.isfinite(point[k])
                or not 0 <= point[k] <= 1
                for k in ("x", "y")
            ):
                reason = "coordinates_invalid"
            if (
                reason is None
                and "z" in point
                and (
                    type(point["z"]) not in (int, float)
                    or not math.isfinite(point["z"])
                )
            ):
                reason = "coordinates_invalid"
            visibility, presence = point.get("visibility"), point.get(
                "presence"
            )
            if reason is None and (
                type(visibility) not in (int, float)
                or not math.isfinite(visibility)
                or not config.min_visibility <= visibility <= 1
            ):
                reason = "visibility_invalid_or_low"
            if (
                reason is None
                and presence is not None
                and (
                    type(presence) not in (int, float)
                    or not math.isfinite(presence)
                    or not config.min_presence <= presence <= 1
                )
            ):
                reason = "presence_invalid_or_low"
            if reason:
                self.invalid[ident] = reason
            else:
                self.points[ident] = point

    def require(self, *ids):
        for ident in ids:
            if ident not in self.points:
                raise ValueError(
                    f"joint_{ident}_" + self.invalid.get(ident, "missing")
                )
        return [self.points[i] for i in ids]

    def xy(self, ident):
        point = self.require(ident)[0]
        return point["x"] * self.size[0] / self.size[1], point["y"]

    def distance(self, a, b):
        return math.dist(self.xy(a), self.xy(b))

    def angle(self, a, b, c):
        # 기존 각도 함수만 재사용하고 Reps 임계값/분석기 상태는 가져오지 않습니다.
        return knee_angle(*self.require(a, b, c), self.size)

    def choose_side(self, required_indices, fixed=None):
        candidates = []
        for side, ids in SIDES.items():
            if fixed is not None and fixed != side:
                continue
            selected = [ids[i] for i in required_indices]
            if all(i in self.points for i in selected):
                quality = min(
                    min(
                        self.points[i]["visibility"],
                        (
                            self.points[i].get("presence")
                            if self.points[i].get("presence") is not None
                            else 1.0
                        ),
                    )
                    for i in selected
                )
                candidates.append((quality, side))
        if not candidates:
            raise ValueError("no_valid_body_side")
        return max(candidates)[1]

    def arms(self):
        self.require(11, 12, 13, 14, 15, 16, 23, 24)
        shoulder = tuple((a + b) / 2 for a, b in zip(self.xy(11), self.xy(12)))
        hip = tuple((a + b) / 2 for a, b in zip(self.xy(23), self.xy(24)))
        torso = math.dist(shoulder, hip)
        if torso < 0.05:
            raise ValueError("degenerate_torso")
        values = []
        for s, e, w in ((11, 13, 15), (12, 14, 16)):
            upper = self.distance(s, e)
            if upper < 0.025:
                raise ValueError("degenerate_arm")
            sx, sy = self.xy(s)
            ex, ey = self.xy(e)
            wx, wy = self.xy(w)
            values.append(
                dict(
                    elbow=self.angle(s, e, w),
                    height=(sy - wy) / torso,
                    upper_elevation=math.degrees(
                        math.atan2(sy - ey, abs(ex - sx))
                    ),
                    wrist_above_elbow=(ey - wy) / upper,
                )
            )
        return values

    def body(self, side):
        s, e, w, h, k, a = SIDES[side]
        self.require(s, h, a)
        sx, sy = self.xy(s)
        ax, ay = self.xy(a)
        hx, hy = self.xy(h)
        length = math.hypot(ax - sx, ay - sy)
        if length < 0.15:
            raise ValueError("degenerate_body_axis")
        normal = (-(ay - sy) / length, (ax - sx) / length)
        if normal[1] > 0:
            normal = (-normal[0], -normal[1])
        # 화면 위쪽이 양수: 골반 상승/처짐의 방향을 좌우 미러링과 무관하게 구분합니다.
        deviation = ((hx - sx) * normal[0] + (hy - sy) * normal[1]) / length
        return dict(
            deviation=deviation,
            alignment=self.angle(s, h, a),
            tilt=math.degrees(math.atan2(abs(ay - sy), abs(ax - sx))),
            length=length,
        )
