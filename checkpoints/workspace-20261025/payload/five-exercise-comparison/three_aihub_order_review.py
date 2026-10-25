"""선택한 AI-Hub 푸시업의 카메라 B 16장 순서를 확대 없이 배열한다."""

import pathlib

from PIL import Image, ImageDraw

import audit
import three_protocol
from resources import check_resources

if __name__ == "__main__":
    root = three_protocol.OUT
    rows = audit.read_json(root / "aihub_pushup_split_plan.json")["records"]
    records = []
    for row in rows:
        check_resources()
        paths = sorted(p["path"] for p in row["images"] if "_B/" in p["path"])
        canvas = Image.new("RGB", (1920, 4 * 300 + 30), "#14212a")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 6),
            row["label"] + " | CAMERA B / ORDER ONLY / FPS UNKNOWN",
            fill="white",
        )
        for j, path in enumerate(paths):
            image = Image.open(root / "sources/aihub_pushup_images" / path)
            image.thumbnail((480, 270))
            x, y = j % 4 * 480, 30 + j // 4 * 300
            canvas.paste(image, (x, y))
            draw.text(
                (x + 4, y + 275), pathlib.PurePosixPath(path).stem, fill="white"
            )
        target = (
            root
            / "review"
            / (
                "aihub_order_"
                + pathlib.PurePosixPath(row["label"]).stem
                + ".jpg"
            )
        )
        canvas.save(target, quality=95)
        records.append(
            dict(
                label=row["label"],
                person=row["person"],
                split=row["split"],
                view="B",
                evidence_image=str(target),
                directly_reviewed=False,
            )
        )
    audit.save_json(
        root / "aihub_order_review_preparation.json",
        dict(records=records, score_labels_created=0),
    )
