"""새 보유 자료 초안에서 관측한 정면 구간의 Heavy 관절 캐시를 준비한다.

측면으로 이름을 바꾸거나 기존 모델 입력에 섞지 않는다. 점수·직접 검토
판정을 생성하지 않으며 다음 구도별 학습 입력 개발을 위한 좌표만 저장한다.
"""

import json

import audit
import extract
from resources import check_resources
from three_existing_review import ROOT


def run() -> dict:
    """직접 라벨을 작성한 활성 3종 구간의 정면 좌표를 저부하로 추출한다.

    Returns:
        dict: 원본 해시·정확한 구간·정면 캐시와 완료 여부. 좌표 추출은
        학습 완료나 총점 평가 가능을 의미하지 않는다.

    Raises:
        OSError: 원본·결과 읽기/쓰기 실패.
        ValueError: 관절 추출 입력·FPS·프레임 계약 불일치.
        RuntimeError: 전원·온도·메모리 보호 위반 또는 Heavy 초기화 실패.
    """
    rows = audit.read_json(ROOT / "score_annotations.json")
    report = dict(
        records=[],
        completed=False,
        extractor="MediaPipe Pose Heavy",
        landmarks_are_not_score_labels=True,
        active_models_updated=False,
    )
    for row in rows:
        if row["exercise"] not in ("pushups", "squats", "shoulder"):
            continue
        check_resources()
        view = "primary" if row["exercise"] == "shoulder" else "front"
        record = dict(
            row,
            **row["assessment"],
            required_views=row["required_input_views"],
            front_sha256=row["video_sha256"]
        )
        receipt = extract.extract_view(
            record, view, row["video_path"], destination=ROOT / "heavy"
        )
        report["records"].append(
            dict(
                id=row["id"],
                exercise=row["exercise"],
                person=row["person"],
                split=row["split"],
                view=row["required_input_views"][0],
                segment=row["segment"],
                extraction=receipt,
                input_status=row["input_status"],
            )
        )
        audit.save_json(ROOT / "heavy_preparation.json", report)
    report["completed"] = True
    audit.save_json(ROOT / "heavy_preparation.json", report)
    return dict(extracted=len(report["records"]), completed=True)


if __name__ == "__main__":
    print(json.dumps(run()))
