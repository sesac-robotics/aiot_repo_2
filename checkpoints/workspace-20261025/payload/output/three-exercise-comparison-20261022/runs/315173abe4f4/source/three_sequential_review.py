"""외부 원본을 처음부터 연속 디코딩하여 탐색 의존 없는 검토판을 만든다.

임의 탐색 검토판과 같은 표본 번호를 사용해 픽셀 차이를 확인한다. 이 도구는
관절·점수를 계산하지 않으며 새 판을 직접 읽기 전에는 라벨로 사용하지 않는다.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw

import audit
import three_protocol
from resources import check_resources


def run() -> dict:
    """긴 외부 후보 세 개의 전체 개요와 문제 시간대를 연속 검토판으로 만든다.

    Returns:
        dict: 원본 표본 번호·검토판·반복 탐색과의 픽셀 불일치 수.
        모든 프레임은 디코딩하지만 직접 검토는 선정된 표본에만 해당한다.

    Raises:
        OSError: 원본·검토판·결과 접근 실패.
        ValueError: 연속 디코딩 또는 원본 메타데이터 실패.
        RuntimeError: 온도·전원·메모리 보호 위반.
    """
    root = three_protocol.OUT
    results = []
    for id, window in [
        ("0xbZWGGEK3Q", (76, 86)),
        ("tH1C1J-0n3w", (180, 206)),
        ("OOe_HrNnQWw", (85, 121)),
    ]:
        source = root / f"sources/youtube_{id}.mp4"
        cap = cv2.VideoCapture(
            str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        probe = cv2.VideoCapture(
            str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        fps, count = cap.get(5), int(cap.get(7))
        groups = [
            np.linspace(0, count - 1, 48).round().astype(int),
            np.linspace(round(window[0] * fps), round(window[1] * fps), 48)
            .round()
            .astype(int),
        ]
        canvases = [Image.new("RGB", (1536, 1792), "#14212a") for _ in groups]
        draws = [ImageDraw.Draw(c) for c in canvases]
        for draw in draws:
            draw.text((8, 7), id + " | FROM FRAME ZERO, NO SEEK", fill="white")
        wanted = set(int(i) for g in groups for i in g)
        errors = []
        try:
            for index in range(count):
                if index % 250 == 0:
                    check_resources()
                if not cap.grab():
                    raise ValueError("Sequential decode failed")
                if index not in wanted:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError("Sequential retrieve failed")
                probe.set(1, index)
                ok2, other = probe.read()
                error = (
                    float(
                        np.max(
                            np.abs(
                                frame.astype(np.int16) - other.astype(np.int16)
                            )
                        )
                    )
                    if ok2
                    else None
                )
                errors.append(
                    dict(frame=index, random_seek_max_pixel_error=error)
                )
                for group, canvas, draw in zip(groups, canvases, draws):
                    if index not in group:
                        continue
                    j = int(np.flatnonzero(group == index)[0])
                    image = Image.fromarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )
                    image.thumbnail((256, 192))
                    x, y = j % 6 * 256, 32 + j // 6 * 220
                    canvas.paste(image, (x + (256 - image.width) // 2, y))
                    draw.text(
                        (x + 3, y + 196),
                        f"f{index} {index/fps:.3f}s",
                        fill="white",
                    )
        finally:
            cap.release()
            probe.release()
        paths = []
        for suffix, canvas, group in zip(
            ["overview", "detail"], canvases, groups
        ):
            output = root / f"review/sequential_{id}_{suffix}.jpg"
            canvas.save(output, quality=96)
            paths.append(str(output))
            audit.save_json(
                output.with_suffix(".json"),
                dict(
                    video_path=str(source),
                    video_sha256=audit.digest(source),
                    fps=fps,
                    sample_frames=[int(v) for v in group],
                    evidence_image=str(output),
                    decoding="from_frame_zero_sequential_no_seek",
                    status="automatically_prepared_not_directly_reviewed",
                ),
            )
        results.append(
            dict(
                id=id,
                source_sha256=audit.digest(source),
                sheets=paths,
                frame_checks=errors,
                random_seek_mismatches=sum(
                    e["random_seek_max_pixel_error"] != 0 for e in errors
                ),
            )
        )
        audit.save_json(
            root / "sequential_review_verification.json",
            dict(records=results, completed=False),
        )
    output = dict(records=results, completed=True, score_generated=False)
    audit.save_json(root / "sequential_review_verification.json", output)
    return output


if __name__ == "__main__":
    print([(r["id"], r["random_seek_mismatches"]) for r in run()["records"]])
