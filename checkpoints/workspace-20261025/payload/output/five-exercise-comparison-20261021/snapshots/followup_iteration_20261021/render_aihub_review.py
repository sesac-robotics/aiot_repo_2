"""선택한 AI-Hub 이미지의 디코딩을 검사하고 구도 검토판을 만든다.

제공자 정상 조건만 구도 선별에 쓰며 자세 정답·점수·학습 가능성은 만들지
않는다. 카메라 A~E는 물리 방향을 직접 확인하기 전 원래 토큰으로 표시한다.
"""

import collections
import pathlib
import cv2
import numpy as np
import audit
from resources import check_resources


def run() -> dict:
    """추출 이미지 전부를 디코딩하고 정상 조건의 사람별 구도판을 저장한다.

    Returns:
        dict: 디코딩 수·해상도·검토판 경로. 직접 검토 여부는 False다.

    Raises:
        ValueError: 이미지 SHA-256 또는 디코딩 실패.
        OSError: 입력·출력 파일 접근 실패.
        RuntimeError: 자원 보호 조건 초과.
    """
    folder = audit.OUT / "aihub_documents"
    selection = audit.read_json(folder / "raw_review_selection.json")
    inspection = audit.read_json(folder / "raw_archive_inspection.json")
    shape_counts = collections.Counter()
    for index, record in enumerate(inspection["image_files"]):
        if index % 20 == 0:
            check_resources()
        path = audit.OUT / "aihub_raw_review" / record["path"]
        if audit.digest(path) != record["sha256"]:
            raise ValueError("Extracted image changed")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("JPEG decode failed")
        shape_counts[str((image.shape[1], image.shape[0]))] += 1
    sheets = []
    for row in selection["selections"]:
        if not row["provider_all_conditions_true"]:
            continue
        check_resources()
        groups = collections.defaultdict(list)
        for name in row["images"]:
            camera = pathlib.PurePosixPath(name).parent.name.rsplit("_", 1)[-1]
            groups[camera].append(name)
        canvas = np.full((660, 1600, 3), 245, dtype=np.uint8)
        kind = "plank" if row["exercise"] == "플랭크" else "Y-Exercise"
        person = "_".join(row["person_tokens"])
        cv2.putText(
            canvas,
            f"{kind} {person} / provider all-true / no score",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
        )
        for column, (camera, names) in enumerate(sorted(groups.items())):
            for time_index, name in enumerate(sorted(names)):
                image = cv2.imread(str(audit.OUT / "aihub_raw_review" / name))
                resized = cv2.resize(
                    image, (320, 180), interpolation=cv2.INTER_AREA
                )
                x, y = column * 320, 70 + time_index * 195
                canvas[y : y + 180, x : x + 320] = resized
                frame = pathlib.PurePosixPath(name).stem.rsplit("-", 1)[-1]
                cv2.putText(
                    canvas,
                    f"{camera} f{frame}",
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    1,
                )
        target = audit.OUT / "aihub_review_sheets" / f"{kind}_{person}.jpg"
        target.parent.mkdir(exist_ok=True)
        if not cv2.imwrite(str(target), canvas):
            raise OSError("Review sheet write failed")
        sheets.append(
            dict(
                exercise=row["exercise"],
                person_tokens=row["person_tokens"],
                label=row["label"],
                image=str(target),
                directly_reviewed=False,
                source_images=row["images"],
            )
        )
    report = dict(
        decoded_images=len(inspection["image_files"]),
        source_SHA256_verified=True,
        shape_counts=dict(shape_counts),
        sheets=sheets,
        images_directly_reviewed=0,
        score_labels_created=0,
        Heavy_extractions=0,
    )
    audit.save_json(folder / "raw_decode_and_sheets.json", report)
    return report


if __name__ == "__main__":
    report = run()
    print(dict(decoded=report["decoded_images"], sheets=len(report["sheets"])))
