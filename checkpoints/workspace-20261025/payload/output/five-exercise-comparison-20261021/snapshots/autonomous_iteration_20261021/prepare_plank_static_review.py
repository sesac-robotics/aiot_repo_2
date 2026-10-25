"""AI-Hub 플랭크 정지 항목 검토용 원본 크롭과 중립 식별자를 준비한다.

촬영 간격이 미확인인 자료에서 3초 지속을 추정하지 않는다. 제공자 조건과
Heavy·규칙 결과를 검토판에 표시하지 않지만 메타데이터에 이미 노출된 개발
검토이므로 전문가 눈가림 평가로 표현하지 않는다.
"""

import collections
import pathlib
import cv2
import numpy as np
import audit
from resources import check_resources


def run() -> dict:
    """B 카메라 f17 원본을 고정 크롭한 7개 검토판과 입력 목록을 저장한다.

    Returns:
        dict: 이미지 SHA-256·잠정 참여자·검토판. 점수 초안은 만들지 않는다.

    Raises:
        ValueError: 이미지 경로·해상도·디코딩이 예상과 다른 경우.
        OSError: 이미지·JSON 읽기 또는 저장 실패.
        RuntimeError: 자원 보호 검사 실패.
    """
    source = audit.read_json(
        audit.OUT / "aihub_documents/raw_review_selection.json"
    )
    rows = []
    for row in source["selections"]:
        if row["exercise"] != "플랭크":
            continue
        images = [
            p for p in row["images"] if "_B/" in p and p.endswith("0000017.jpg")
        ]
        if len(images) != 1:
            raise ValueError("Expected one B frame17")
        path = audit.OUT / "aihub_raw_review" / images[0]
        identifier = pathlib.PurePosixPath(images[0]).parent.name.removesuffix(
            "_B"
        )
        rows.append(
            dict(
                id=identifier,
                person_token=row["person_tokens"][0],
                image_path=str(path),
                image_sha256=audit.digest(path),
                original_label=row["label"],
                camera="B",
                frame_number=17,
                source_provider_split="Validation",
                evaluation_role="exposed_development",
                label_before_new_Heavy_or_rule=True,
                FPS_confirmed=False,
            )
        )
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row["person_token"]].append(row)
    sheets = []
    for person, group in sorted(grouped.items()):
        check_resources()
        group.sort(key=lambda r: r["id"])
        canvas = np.full((1980, 1400, 3), 245, dtype=np.uint8)
        for i, row in enumerate(group):
            image = cv2.imread(row["image_path"])
            if image is None or image.shape[:2] != (1080, 1920):
                raise ValueError("Unexpected source image")
            patch = image[300:950, 450:1450]
            patch = cv2.resize(patch, (700, 455), interpolation=cv2.INTER_AREA)
            x, y = (i % 2) * 700, (i // 2) * 490 + 30
            canvas[y : y + 455, x : x + 700] = patch
            cv2.putText(
                canvas,
                row["id"],
                (x + 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
            )
        path = audit.OUT / "plank_static_review" / (person + ".jpg")
        path.parent.mkdir(exist_ok=True)
        if not cv2.imwrite(str(path), canvas):
            raise OSError("Review image write failed")
        sheets.append(
            dict(
                person=person,
                path=str(path),
                ids=[r["id"] for r in group],
                directly_reviewed=False,
                crop_xyxy=[450, 300, 1450, 950],
            )
        )
    report = dict(
        records=rows,
        sheets=sheets,
        score_labels_created=0,
        protocol_ko="B 사선 구도의 f17 정지 항목 개발 검토. 화면 정렬·팔꿈치 전후 위치만 후보. 밴드 추가 지지와 시간 지속은 별도 미확인.",
        expert_validated=False,
        independent_final_test=False,
    )
    audit.save_json(audit.OUT / "plank_static_review/inputs.json", report)
    return report


if __name__ == "__main__":
    report = run()
    print(dict(images=len(report["records"]), sheets=len(report["sheets"])))
