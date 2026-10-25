"""동일 원본 프레임에 원본·크롭 Heavy 관절을 되돌려 그린다.

시간 이동·확대에 따른 시각적 착시를 줄이는 개발 진단이다. 관절 위치의
해부학적 정답이나 새 자세 라벨을 생성하지 않는다.
"""

import cv2
import numpy as np
import audit
from resources import check_resources


def run() -> dict:
    """개선·관측 탈락 사례의 같은 시점 관절 비교 PNG를 저장한다.

    Returns:
        dict: 이미지·프레임·좌표 역변환 기록. 직접 검토 여부는 False다.

    Raises:
        OSError: 입력·캐시 또는 출력 접근 실패.
        ValueError: 원본 프레임 디코딩 실패.
        RuntimeError: 자원 보호 중단.
    """
    direct = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "w_targeted_direct_review.json")[
            "reviews"
        ]
    }
    variants = {
        name: {
            r["id"]: r for r in audit.read_json(audit.OUT / filename)["results"]
        }
        for name, filename in (
            ("upper_body", "w_roi_trial_results.json"),
            ("head_to_knee", "w_roi_context_results.json"),
        )
    }
    checks = []
    for identifier in ("qevd_00239387", "qevd_00101824"):
        check_resources()
        row = direct[identifier]
        with np.load(
            audit.OUT / "w_targeted_cache" / identifier / "primary.npz"
        ) as data:
            frames, baseline = data["frame_numbers"], data["image_xy"]
        core = row["direct_W_core_frames"]
        index = int(np.argmin(abs(frames - sum(core) / 2)))
        frame = int(frames[index])
        cap = cv2.VideoCapture(
            row["video_path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        cap.set(1, frame)
        ok, image = cap.read()
        cap.release()
        if not ok:
            raise ValueError("Cannot decode source frame")
        points = {"original": baseline[index]}
        for name, cache in (
            ("upper_body", "w_roi_cache"),
            ("head_to_knee", "w_roi_context_cache"),
        ):
            x, y, _, _ = variants[name][identifier]["crop_xywh"]
            with np.load(
                audit.OUT / cache / (identifier + "_roi") / "primary.npz"
            ) as data:
                points[name] = data["image_xy"][index] + [x, y]
        panels = []
        for name, xy in points.items():
            panel = image.copy()
            for a, b in (
                (11, 12),
                (11, 13),
                (13, 15),
                (12, 14),
                (14, 16),
                (11, 23),
                (12, 24),
                (23, 24),
            ):
                if np.isfinite(xy[[a, b]]).all():
                    cv2.line(
                        panel,
                        tuple(xy[a].astype(int)),
                        tuple(xy[b].astype(int)),
                        (40, 240, 100),
                        2,
                    )
            for joint in (11, 12, 13, 14, 15, 16, 23, 24):
                if np.isfinite(xy[joint]).all():
                    point = tuple(xy[joint].astype(int))
                    cv2.circle(panel, point, 4, (0, 220, 255), -1)
                    cv2.putText(
                        panel,
                        str(joint),
                        point,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        1,
                    )
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (20, 20, 20), -1)
            cv2.putText(
                panel,
                f"{name} f{frame}",
                (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
            panels.append(panel)
        target = audit.OUT / "w_roi_context" / (identifier + "_same_frame.png")
        if not cv2.imwrite(str(target), np.hstack(panels)):
            raise OSError("PNG write failed")
        checks.append(
            dict(
                id=identifier,
                frame=frame,
                image=str(target),
                ROI_points_mapped_to_source_pixels=True,
                directly_reviewed=False,
            )
        )
    report = dict(records=checks, independent_final_test=False)
    audit.save_json(audit.OUT / "w_roi_visual_checks.json", report)
    return report


if __name__ == "__main__":
    print(run())
