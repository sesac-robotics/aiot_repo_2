"""직접 확인한 촬영 잘림을 두 채점 방식의 공통 관절 입력에 반영한다.

수동 촬영 메타데이터이며 자동 잘림 검출기가 아니다. 참조 등급·점수는
읽지 않고 원본 해시·구간·시점이 모두 일치한 관절만 결측 처리한다.
"""

import numpy as np

import audit
import three_protocol


def apply_framing(
    sequence: np.ndarray, manifest: dict, view: str
) -> np.ndarray:
    """촬영 경계 밖으로 확인한 관절을 원본 사본에서 제거한다.

    Args:
        sequence: N×33×2 원본 픽셀 좌표. 입력은 변경하지 않는다.
        manifest: 영상 ID·SHA256·프레임 구간이 있는 입력 명세.
        view: side, front 또는 primary 촬영 시점.

    Returns:
        np.ndarray: 직접 관측 불가한 관절을 NaN으로 바꾼 좌표 사본.
        해당 입력의 수동 메타데이터가 없으면 좌표를 그대로 복사한다.

    Raises:
        OSError: 촬영 메타데이터 파일 읽기 실패.
        ValueError: 원본 해시·구간 또는 입력 좌표 형상 불일치.
    """
    if sequence.ndim != 3 or sequence.shape[1:] != (33, 2):
        raise ValueError("N x 33 x 2 coordinates required")
    policy = audit.read_json(three_protocol.OUT / "input_observability.json")
    result = sequence.copy()
    for row in policy["records"]:
        if row["id"] != manifest["id"] or row["view"] != view:
            continue
        if row["video_sha256"] != manifest["video_sha256"] or any(
            row["segment"][key] != manifest["segment"][key]
            for key in ("start_frame", "end_frame", "fps")
        ):
            raise ValueError("Manual framing metadata source mismatch")
        result[:, row["unobservable_joints"]] = np.nan
    return result
