"""W 등급 불일치를 투영·관절 위치·채점 계약으로 분리해 진단한다.

수동 점은 원본 표본을 본 비전문가의 대략적인 화면 위치이며 정답이 아니다.
기존 등급을 알고 진행한 개발 재검토로 독립·눈가림 검토가 아니다.
"""

import itertools

import cv2
import numpy as np

import audit
import geometry
from resources import check_resources

# 원본 PNG만 표시한 뒤 기록한 왼쪽·오른쪽 어깨/팔꿈치/손목 화면 좌표.
MANUAL = {
    "qevd_00015905": [
        [[400, 186], [433, 153], [456, 122]],
        [[307, 191], [270, 160], [243, 124]],
    ],
    "qevd_00014884": [
        [[332, 155], [326, 137], [326, 117]],
        [[307, 155], [309, 134], [300, 122]],
    ],
    "qevd_00035657": [
        [[334, 158], [365, 185], [357, 147]],
        [[282, 154], [245, 177], [245, 135]],
    ],
    "qevd_00196196": [
        [[354, 182], [362, 196], [385, 175]],
        [[253, 180], [238, 196], [210, 175]],
    ],
    "qevd_00265931": [
        [[356, 102], [379, 118], [414, 103]],
        [[304, 95], [283, 95], [272, 65]],
    ],
    "qevd_00265212": [
        [[375, 108], [395, 145], [394, 91]],
        [[274, 108], [250, 141], [252, 87]],
    ],
}


def projection_counterexample() -> dict:
    """같은 2D 두 벡터가 서로 다른 3D 각도를 가지는 반례를 반환한다.

    Returns:
        dict: 임의 길이 단위의 두 3D 벡터쌍, 동일한 XY, 도 단위 각도.
        실제 영상의 깊이를 추정하거나 자세 등급을 생성하지 않는다.

    Raises:
        AssertionError: 직교 투영 반례의 수학 조건이 성립하지 않는 경우.
    """
    xy = np.array([[1.0, 0.0], [-np.sqrt(3) / 2, 0.5]])
    flat = np.column_stack([xy, [0.0, 0.0]])
    lifted = np.column_stack([xy, [1.0, np.sqrt(3) / 2]])
    angle = lambda x: float(
        np.degrees(
            np.arccos(
                np.clip(
                    np.dot(x[0], x[1])
                    / (np.linalg.norm(x[0]) * np.linalg.norm(x[1])),
                    -1,
                    1,
                )
            )
        )
    )
    assert np.array_equal(flat[:, :2], lifted[:, :2])
    assert abs(angle(lifted) - 90) < 1e-8
    return dict(
        projected_angle_deg=angle(xy),
        flat_3d_angle_deg=angle(flat),
        lifted_3d_angle_deg=angle(lifted),
        vectors_flat=flat.tolist(),
        vectors_lifted=lifted.tolist(),
        inference_ko="단일 화면 각도 150도는 실제 3D 150도와 90도를 구별하지 못한다. 실제 영상의 원인을 확정하는 반례는 아니다.",
    )


def run() -> dict:
    """6개 기존 개발 자료의 수동·Heavy 화면 관절 진단과 그림을 저장한다.

    Returns:
        dict: 프레임 차이, 수동 점의 ±8픽셀 상자 민감도, Heavy 값,
        계약 불일치와 투영 반례. 점수 라벨·AI 학습 입력은 변경하지 않는다.

    Raises:
        OSError: 원본 장면·캐시·메타데이터 읽기 또는 저장 실패.
        ValueError: 영상 해시가 다르거나 프레임 차이가 1을 넘는 경우.
        RuntimeError: 자원 보호 조건 충족.
    """
    rows = []
    labels = {
        row["id"]: row for row in audit.read_json(audit.OUT / "labels.json")
    }
    for identifier, points in MANUAL.items():
        check_resources()
        folder = audit.OUT / "w_review"
        path = next(folder.glob(identifier + "*_raw.json"))
        source = audit.read_json(path)
        metadata = audit.read_json(
            audit.OUT / "heavy" / identifier / "primary.json"
        )
        if source["video_sha256"] != metadata["config"]["source_sha256"]:
            raise ValueError("Source mismatch")
        with np.load(audit.OUT / "heavy" / identifier / "primary.npz") as cache:
            nearest = int(
                np.argmin(abs(cache["frame_numbers"] - source["frame"]))
            )
            frame = int(cache["frame_numbers"][nearest])
            xy = cache["image_xy"][nearest]
        if abs(frame - source["frame"]) > 1:
            raise ValueError("Manual frame and extraction too far apart")
        pixels = cv2.imread(source["image"])
        sides = []
        for side, indices, manual in zip(
            ("left", "right"), ((11, 13, 15), (12, 14, 16)), points
        ):
            manual = np.asarray(manual, dtype=float)
            heavy = xy[list(indices)]
            measurement = lambda p: float(
                geometry.angle_deg(p[:1], p[1:2], p[2:])[0]
            )
            variants = [
                measurement(manual + np.array(offset).reshape(3, 2))
                for offset in itertools.product((-8, 8), repeat=6)
            ]
            manual_angle = measurement(manual)
            heavy_angle = measurement(heavy)
            sides.append(
                dict(
                    side=side,
                    manual_xy=manual.tolist(),
                    manual_projected_deg=manual_angle,
                    manual_pixel_box_radius=8,
                    corner_sensitivity_min_max_deg=[
                        min(variants),
                        max(variants),
                    ],
                    heavy_xy=heavy.tolist(),
                    heavy_projected_deg=(
                        heavy_angle if np.isfinite(heavy_angle) else None
                    ),
                    manual_to_heavy_joint_distance_px=np.linalg.norm(
                        manual - heavy, axis=1
                    ).tolist(),
                )
            )
            for coords, color in (
                (manual, (0, 210, 255)),
                (heavy, (255, 170, 0)),
            ):
                if not np.isfinite(coords).all():
                    continue
                p = coords.round().astype(int)
                cv2.polylines(pixels, [p], False, color, 2)
                for point in p:
                    cv2.circle(pixels, tuple(point), 4, color, -1)
        cv2.putText(
            pixels,
            "Yellow: manual approx | Blue: Heavy nearby frame",
            (8, pixels.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        output = folder / (identifier + "_manual_heavy.png")
        cv2.imwrite(str(output), pixels)
        rows.append(
            dict(
                id=identifier,
                direct_raw_frame=source["frame"],
                nearest_heavy_frame=frame,
                same_frame=frame == source["frame"],
                sides=sides,
                evidence_image=str(output),
                label_draft_levels=labels[identifier]["criterion_levels"],
                label_draft_note=labels[identifier].get("note_ko"),
                anatomical_ground_truth=False,
                independent_blind_review=False,
            )
        )
    result = dict(
        cases=rows,
        projection_counterexample=projection_counterexample(),
        rubric_contract_ko="라벨 기준은 양 팔꿈치 굽힘과 손을 위쪽에 두는 W 모양이다. 화면 각도 90±20도는 기준에 없으며 이전 규칙이 추가한 미검증 조건이다.",
        root_cause_status=dict(
            phase_selection="separately improved, not sufficient",
            rubric_numeric_target_mismatch="confirmed_by_code_and_rubric",
            projection_non_identifiability="mathematically_demonstrated_not_per_clip_causality",
            landmark_error="approximate_2D_review_only_not_anatomical_truth",
            draft_label_error="not_resolved_without_independent_review",
        ),
        score_labels_modified=False,
        independent_final_test=False,
    )
    audit.save_json(audit.OUT / "ytw_w_root_cause.json", result)
    return result


if __name__ == "__main__":
    run()
