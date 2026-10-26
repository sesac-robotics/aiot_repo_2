"""출처가 확인된 외부 플랭크 후보를 Heavy 관절과 다섯 시점으로 선별한다.

자동 선별·검토판 생성만 수행하며 점수·정상 오류 라벨을 만들지 않는다.
한 장면의 통과는 연속 구간 적격성이나 참여자 독립성의 확인이 아니다.
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw

import audit
from extract import MODEL
from resources import check_resources
import geometry


def run(delegate: str = "GPU") -> list[dict]:
    """두 신규 외부 후보의 관절·표본 검토판·구도 진단을 저장한다.

    Args:
        delegate: GPU 또는 CPU. CPU는 GPU 선별 로그의 교차 확인용이다.

    Returns:
        list[dict]: 원본 해시·표본 프레임·관측 관절·팔꿈치 각도 도·구도 후보.
        후보 여부는 직접 검토 우선순위일 뿐 학습 적격 판정이 아니다.

    Raises:
        OSError: 원본·출처 메타데이터 읽기 또는 결과 저장 실패.
        ValueError: 출처 해시·권리 표시·영상 메타데이터·디코딩 오류.
        RuntimeError: GPU 추출 또는 자원 보호 조건 위반.
    """
    if delegate not in ("GPU", "CPU"):
        raise ValueError("Unknown inference delegate")
    folder = audit.OUT / (
        "external_plank_screen"
        if delegate == "GPU"
        else "external_plank_screen_cpu_check"
    )
    folder.mkdir(exist_ok=True)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL),
            delegate=getattr(mp.tasks.BaseOptions.Delegate, delegate),
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    results = []
    check_resources()
    with mp.tasks.vision.PoseLandmarker.create_from_options(
        options
    ) as detector:
        for ident in [753559, 873358]:
            check_resources()
            path = audit.OUT / "sources" / f"dvids_{ident}.mp4"
            source = audit.read_json(
                path.with_name(f"dvids_{ident}_acquisition.json")
            )
            if (
                not source["public_domain_marker"]
                or audit.digest(path) != source["sha256"]
            ):
                raise ValueError("Unverified candidate source")
            cap = cv2.VideoCapture(
                str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
            )
            fps = cap.get(5)
            count = int(cap.get(7))
            if fps <= 0 or count < 5:
                raise ValueError("Invalid video")
            indices = np.rint(
                (count - 1) * np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            ).astype(int)
            canvas = Image.new("RGB", (1536, 650), "#14212a")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (8, 8),
                f"DVIDS {ident} | AUTOMATIC SAMPLES - NO SCORE",
                fill="white",
            )
            seq = []
            frames = []
            try:
                for j, index in enumerate(indices):
                    check_resources()
                    cap.set(1, int(index))
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError("Candidate decode failed")
                    h, w = frame.shape[:2]
                    scale = min(1, 640 / max(h, w))
                    small = cv2.resize(
                        frame, (round(w * scale), round(h * scale))
                    )
                    result = detector.detect(
                        mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(
                                cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                            ),
                        )
                    )
                    points = np.full((33, 2), np.nan)
                    if result.pose_landmarks:
                        for k, point in enumerate(result.pose_landmarks[0]):
                            if (
                                point.visibility >= 0.5
                                and point.presence >= 0.5
                                and 0 <= point.x <= 1
                                and 0 <= point.y <= 1
                            ):
                                points[k] = point.x * w, point.y * h
                    seq.append(points)
                    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    im.thumbnail((500, 270))
                    x = (j % 3) * 512
                    y = 32 + (j // 3) * 305
                    canvas.paste(im, (x, y))
                    draw.text(
                        (x + 4, y + 274),
                        f"f{index} | {index/fps:.2f}s",
                        fill="white",
                    )
                    sides = []
                    for sh, el, wr, hip, knee, ank, toe in [
                        (11, 13, 15, 23, 25, 27, 31),
                        (12, 14, 16, 24, 26, 28, 32),
                    ]:
                        chain = np.isfinite(
                            points[[sh, el, wr, hip, knee, ank, toe]]
                        ).all()
                        torso = np.linalg.norm(points[sh] - points[hip])
                        span = (
                            float(abs(points[sh, 0] - points[ank, 0]) / torso)
                            if chain and torso > 10
                            else None
                        )
                        elbow = (
                            float(
                                geometry.angle_deg(
                                    points[[sh]], points[[el]], points[[wr]]
                                )[0]
                            )
                            if chain
                            else None
                        )
                        sides.append(
                            dict(
                                full_chain_observed=bool(chain),
                                horizontal_span=span,
                                elbow_angle_deg=elbow,
                                candidate=bool(
                                    chain and span >= 1.5 and 45 <= elbow <= 135
                                ),
                            )
                        )
                    frames.append(
                        dict(
                            frame=int(index),
                            seconds=float(index / fps),
                            side_proxies=sides,
                        )
                    )
            finally:
                cap.release()
            image_path = folder / f"dvids_{ident}.jpg"
            canvas.save(image_path, quality=95)
            np.savez_compressed(
                folder / f"dvids_{ident}.npz",
                image_xy=np.asarray(seq),
                frame_numbers=indices,
            )
            results.append(
                dict(
                    id=f"dvids_{ident}",
                    delegate=delegate,
                    source_url=source["source_url"],
                    video_path=str(path),
                    video_sha256=source["sha256"],
                    fps=fps,
                    samples=frames,
                    evidence_image=str(image_path),
                    candidate_for_direct_review=any(
                        side["candidate"]
                        for frame in frames
                        for side in frame["side_proxies"]
                    ),
                    direct_visual_reviewed=False,
                    score_labels_created=0,
                    independent_final_test=False,
                )
            )
            audit.save_json(folder / "results.json", results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delegate", choices=["GPU", "CPU"], default="GPU")
    print(run(parser.parse_args().delegate))
