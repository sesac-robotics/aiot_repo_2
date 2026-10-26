"""외부 영상의 직접 검토로 고른 시간대를 상세 시간순 검토판으로 준비한다.

자동 준비는 점수 라벨링이 아니다. 정상·오류 자막은 점수로 변환하지 않으며
각 장면과 목표 운동 변형·시점 일치를 직접 검토한 뒤에만 학습 자료로 쓴다.
"""

import audit
import external_review
import three_protocol
from resources import check_resources

WINDOWS = [
    ("OOe_HrNnQWw", 85, 121),
    ("0xbZWGGEK3Q", 45, 82),
    ("tH1C1J-0n3w", 181, 206),
    ("tH1C1J-0n3w", 24, 35),
]


def run() -> dict:
    """사전 선택한 시간대의 48장 표본과 원본 프레임 연결을 저장한다.

    Returns:
        dict: 원본 ID·구간·검토판·프레임 번호. 직접 등급은 포함하지 않는다.

    Raises:
        OSError: 원본·결과 접근 실패.
        ValueError: 영상 구간 또는 디코딩 실패.
        RuntimeError: 온도·전원·메모리 보호 조건 위반.
    """
    root = three_protocol.OUT
    results = []
    for id, start, end in WINDOWS:
        check_resources()
        target = root / f"review/youtube_{id}_{start}_{end}.jpg"
        metadata = target.with_suffix(".json")
        sheet = (
            audit.read_json(metadata)
            if target.exists() and metadata.exists()
            else external_review.contact_sheet(
                root / f"sources/youtube_{id}.mp4", target, start, end, 48
            )
        )
        results.append(
            dict(
                id=id,
                start_seconds=start,
                end_seconds=end,
                sheet=sheet,
                score_generated=False,
            )
        )
    report = dict(records=results, directly_reviewed=False)
    audit.save_json(root / "external_window_preparation.json", report)
    return report


if __name__ == "__main__":
    print(len(run()["records"]))
