"""기존 AI-Hub 원시 분할 TAR에서 푸시업 후보의 순서 이미지와 검토판만 준비한다.

제공자 조건·정상 문자열을 점수로 바꾸지 않는다. 원본 FPS가 미확인인
이미지 순서 자료이므로 영상 시간·처리 속도 정답으로 취급하지 않는다.
"""

import contextlib
import hashlib
import pathlib
import re
import tarfile

from PIL import Image, ImageDraw

import audit
import inspect_aihub_raw
import three_protocol
from resources import check_resources
from segmented_reader import SegmentedReader


def run() -> dict:
    """사전 분할한 네 참여자의 두 제공자 시나리오에서 이미지 640장을 한도로 추출한다.

    Returns:
        dict: 원본 이미지 해시·카메라·순서·검토판·사전 분할. 점수는 생성하지 않는다.

    Raises:
        OSError: 원본·출력 파일 접근 실패.
        ValueError: 원본 변경·분할 페이로드·예산·선택 이미지 누락.
        tarfile.TarError: TAR 읽기 실패.
        RuntimeError: 자원 보호 조건 위반.
    """
    root = three_protocol.OUT
    old = audit.OUT / "aihub_documents"
    inventory = audit.read_json(old / "validation_body_inventory.json")
    roles = {
        "Z56": "train",
        "Z115": "train",
        "Z61": "validation",
        "Z84": "validation",
    }
    rows = [
        r
        for r in inventory["records"]
        if r["exercise"] == "푸시업"
        and r["provider_type"] in ("561", "564")
        and r["person_tokens"][0] in roles
    ]
    plan = dict(
        records=[
            dict(
                label=r["label_member"],
                person=r["person_tokens"][0],
                split=roles[r["person_tokens"][0]],
                images=r["images"],
                provider_description=r["description"],
            )
            for r in rows
        ],
        source_fps_confirmed=False,
        role_assigned_before_new_pushup_pixels=True,
        earlier_plank_person_exposure_possible=True,
        independent_final_test=False,
    )
    audit.save_json(root / "aihub_pushup_split_plan.json", plan)
    selected = {p["path"] for r in rows for p in r["images"]}
    if len(rows) != 8 or len(selected) != 640:
        raise ValueError(
            "Expected eight records with 16 frames and five cameras"
        )
    source = audit.BASE / "data/aihub231/raw/valid_body_download.tar"
    before = source.stat()
    extracted = []
    total = 0
    with contextlib.ExitStack() as stack:
        outer = stack.enter_context(tarfile.open(source, "r:"))
        parts = []
        for member in outer.getmembers():
            match = re.fullmatch(
                r"valid_body_data\.tar\.part([0-9]+)",
                pathlib.PurePosixPath(member.name).name,
            )
            if member.isfile() and match:
                parts.append((int(match[1]), member.offset_data, member.size))
        if sum(p[2] for p in parts) != inspect_aihub_raw.EXPECTED_PAYLOAD_BYTES:
            raise ValueError("Unexpected raw payload")
        stream = stack.enter_context(source.open("rb"))
        joined = stack.enter_context(SegmentedReader(stream, parts))
        payload = stack.enter_context(tarfile.open(fileobj=joined, mode="r:"))
        for index, member in enumerate(payload):
            if index % 500 == 0:
                check_resources()
            if not member.isfile():
                continue
            key = inspect_aihub_raw.canonical_image_path(member.name)
            if key not in selected:
                continue
            total += member.size
            if total > 256 * 1024**2 or member.size > 20 * 1024**2:
                raise ValueError("Selective extraction budget exceeded")
            content = payload.extractfile(member).read()
            target = root / "sources/aihub_pushup_images" / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            extracted.append(
                dict(
                    path=key,
                    local_path=str(target),
                    sha256=hashlib.sha256(content).hexdigest(),
                )
            )
            if len(extracted) == len(selected):
                break
    if len(extracted) != len(selected) or (
        before.st_size,
        before.st_mtime_ns,
    ) != (source.stat().st_size, source.stat().st_mtime_ns):
        raise ValueError("Missing images or changed source")
    sheets = []
    for row in plan["records"]:
        canvas = Image.new("RGB", (1500, 3 * 400 + 30), "#14212a")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 6),
            row["label"] + " | ORDERED IMAGES / FPS UNKNOWN",
            fill="white",
        )
        for j, (frame, view) in enumerate(
            (f, v) for f in (1, 17, 31) for v in "ABCDE"
        ):
            match = next(
                p["path"]
                for p in row["images"]
                if f"_{view}/" in p["path"]
                and p["path"].endswith(f"{frame:07d}.jpg")
            )
            image = Image.open(root / "sources/aihub_pushup_images" / match)
            image.thumbnail((300, 370))
            x, y = j % 5 * 300, 30 + j // 5 * 400
            canvas.paste(image, (x + (300 - image.width) // 2, y))
            draw.text(
                (x + 4, y + 375),
                f"CAM {view} / SOURCE FRAME {frame}",
                fill="white",
            )
        target = (
            root
            / "review"
            / (
                "aihub_pushup_"
                + pathlib.PurePosixPath(row["label"]).stem
                + ".jpg"
            )
        )
        canvas.save(target, quality=94)
        sheets.append(
            dict(
                label=row["label"],
                person=row["person"],
                split=row["split"],
                evidence_image=str(target),
                directly_reviewed=False,
            )
        )
    report = dict(
        images=extracted,
        extracted_bytes=total,
        sheets=sheets,
        source_fps_confirmed=False,
        score_labels_created=0,
        source_archive_bytes=before.st_size,
        original_archive_unchanged=True,
        selected_images_only=True,
    )
    audit.save_json(root / "aihub_pushup_review_preparation.json", report)
    return report


if __name__ == "__main__":
    print(len(run()["images"]))
