"""기존 허용 원본의 미검토 중간 구간에서 오류 시범 후보를 찾는다.

동일 수행자·반복 시범은 새 참여자나 독립 시험으로 계산하지 않는다.
"""

import audit
from external_review import contact_sheet


def run() -> dict:
    """공군 825280 원본 중간 구간 72개 표본을 준비한다.

    Returns:
        dict: 원본 SHA·표본 프레임·근거 이미지 경로. 점수는 생성하지 않는다.

    Raises:
        ValueError: 영상 구간 또는 디코딩 오류.
        RuntimeError: 자원 보호 조건 충족.
        OSError: 원본 읽기·검토판 저장 실패.
    """
    return contact_sheet(
        audit.SOURCE / "sources/forearm_plank.mp4",
        audit.OUT / "review/airforce_825280_middle.jpg",
        30,
        177,
        72,
    )


if __name__ == "__main__":
    print(run())
