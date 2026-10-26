"""허용된 해군 교육 원본의 오류 시범 후보를 시간순으로 준비한다."""

import audit
from external_review import contact_sheet


def run() -> None:
    """100~140초·160~207초 표본을 저장한다. 점수는 생성하지 않는다.

    Returns:
        None: 각 구간 72장 검토판과 원본 프레임 메타데이터를 저장한다.

    Raises:
        ValueError: 구간·영상 디코딩 오류.
        RuntimeError: 자원 보호 중단.
        OSError: 원본 또는 검토판 접근 실패.
    """
    source = audit.OUT / "sources/dvids_773312.mp4"
    for start, end in ((100, 140), (160, 207)):
        contact_sheet(
            source,
            audit.OUT / f"review/navy773312_{start}_{end}.jpg",
            start,
            end,
            72,
        )


if __name__ == "__main__":
    run()
