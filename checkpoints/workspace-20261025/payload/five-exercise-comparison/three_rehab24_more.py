"""초기 양끝 오류 사례 외의 중간 스쿼트 오류 후보를 준비한다.

제공자 문자열은 후보 선정에만 사용한다. 동일 참여자·원본의 기존 개발 분할을
유지하며 관절·규칙·모델 점수를 보지 않고 추가 직접 검토판만 만든다.
"""

import copy
import csv
import pathlib

import audit
import external_review
import three_protocol
from resources import check_resources


def run() -> dict:
    """네 기존 참여자의 중간 반복을 양 구도의 24장 검토판으로 준비한다.

    Returns:
        dict: 원본 식별자·구간·준비된 검토판 목록. 직접 점수는 없다.

    Raises:
        OSError: CSV·원본·결과 파일 접근 실패.
        ValueError: 대응 반복 누락 또는 디코딩 실패.
        RuntimeError: 자원 보호 조건 위반.
    """
    root = three_protocol.OUT
    bases = audit.read_json(root / "rehab24_review_preparation.json")["records"]
    with (root / "Segmentation.csv").open() as file:
        annotations = list(csv.DictReader(file, delimiter=";"))
    output = []
    for person in ["3", "5", "7", "8"]:
        check_resources()
        annotation = next(
            r
            for r in annotations
            if r["exercise_id"] == "6"
            and r["person_id"] == person
            and r["repetition_number"] == "8"
            and r["cam17_orientation"] == "front"
        )
        row = copy.deepcopy(
            next(r for r in bases if r["person"] == "rehab24_" + person)
        )
        row["id"] = f"rehab24_{annotation['video_id']}_rep8"
        row["provider_annotation"] = annotation
        row["source_labels"] = ["correctness=" + annotation["correctness"]]
        row["segment"] = dict(
            start_frame=int(annotation["first_frame"]),
            end_frame=int(annotation["last_frame"]),
            fps=30.0,
        )
        for view, path in [
            ("side", row["video_path"]),
            ("front", row["front_path"]),
        ]:
            target = root / f'review/rehab24/{row["id"]}_{view}.jpg'
            metadata = target.with_suffix(".json")
            row["sheets"][view] = (
                audit.read_json(metadata)
                if target.exists() and metadata.exists()
                else external_review.contact_sheet(
                    pathlib.Path(path),
                    target,
                    row["segment"]["start_frame"] / 30,
                    row["segment"]["end_frame"] / 30,
                    24,
                )
            )
        output.append(row)
        audit.save_json(
            root / "rehab24_additional_preparation.json",
            dict(records=output, completed=False, directly_reviewed=False),
        )
    result = dict(records=output, completed=True, directly_reviewed=False)
    audit.save_json(root / "rehab24_additional_preparation.json", result)
    return result


if __name__ == "__main__":
    print(len(run()["records"]))
