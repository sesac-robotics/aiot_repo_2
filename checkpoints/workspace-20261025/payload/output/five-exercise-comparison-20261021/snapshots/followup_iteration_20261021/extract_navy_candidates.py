"""직접 작성한 해군 플랭크 초안의 관절 관측 가능성을 확인한다."""

import audit
from extract import extract_view


def run() -> dict:
    """두 구간을 공통 Heavy 설정으로 추출한다. 학습 라벨을 생성하지 않는다.

    Returns:
        dict: 구간별 캐시·관측률·처리 초. 같은 사람·원본이며 독립 시험이 아니다.

    Raises:
        ValueError: 원본·구간·캐시 계약 오류.
        RuntimeError: 자원 보호·Heavy 추출 중단.
        OSError: 원본 또는 결과 저장 실패.
    """
    rows = audit.read_json(audit.OUT / "navy773312_direct_drafts.json")[
        "labels"
    ]
    results = []
    for row in rows:
        prepared = dict(row, required_views=["side"])
        results.append(
            dict(
                id=row["id"],
                extraction=extract_view(prepared, "side", row["video_path"]),
            )
        )
    report = dict(
        results=results,
        independent_final_test=False,
        labels_generated_by_rule=False,
    )
    audit.save_json(audit.OUT / "navy773312_extraction.json", report)
    return report


if __name__ == "__main__":
    print(run())
