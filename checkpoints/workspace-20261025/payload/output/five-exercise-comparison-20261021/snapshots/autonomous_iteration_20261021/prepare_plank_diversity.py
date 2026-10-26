"""새 외부 플랭크 두 원본의 시간순 구도 검토 표본을 만든다."""

import audit
from external_review import contact_sheet


def run() -> None:
    """원본 두 개의 전체 범위에서 각각 48장 표본을 저장한다.

    Returns:
        None: 검토판·해시·프레임 번호를 저장한다. 등급은 생성하지 않는다.

    Raises:
        ValueError: 영상 메타데이터·디코딩 오류.
        OSError: 원본 또는 출력 접근 실패.
        RuntimeError: 자원 보호 중단.
    """
    for identifier in ("745041", "859787"):
        contact_sheet(
            audit.OUT / f"sources/dvids_{identifier}.mp4",
            audit.OUT / f"review/dvids_{identifier}_overview.jpg",
            0,
            None,
            48,
        )


if __name__ == "__main__":
    run()
