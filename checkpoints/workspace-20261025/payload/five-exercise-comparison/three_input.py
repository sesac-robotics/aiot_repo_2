"""직접 확인한 촬영 잘림을 두 채점 방식의 공통 관절 입력에 반영한다.

수동 촬영 메타데이터이며 자동 잘림 검출기가 아니다. 참조 등급·점수는
읽지 않고 원본 해시·구간·시점이 모두 일치한 관절만 결측 처리한다.
"""

import copy

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
        result[:, row.get("unobservable_joints", [])] = np.nan
    return result


def apply_observation(
    notes: list[dict], manifest: dict, view: str
) -> list[dict]:
    """직접 확인한 시점의 특정 항목만 보류하고 보이는 다른 항목은 유지한다.

    Args:
        notes: 관절 기하로 계산한 네 항목의 관측·측정 결과.
        manifest: 원본 ID·SHA256·시작 끝 프레임·FPS가 있는 입력 명세.
        view: 해당 주 입력의 side 또는 primary 시점.

    Returns:
        list[dict]: 직접 확인한 관측 제한만 반영한 사본. 등급은 읽지 않는다.
        수동 입력 감사이며 자동 관측 검출 성능으로 해석하지 않는다.

    Raises:
        OSError: 촬영 메타데이터 읽기 실패.
        ValueError: 정책과 입력의 원본·구간 불일치.
    """
    result = copy.deepcopy(notes)
    policy = audit.read_json(three_protocol.OUT / "input_observability.json")
    for row in policy["records"]:
        if row["id"] != manifest["id"] or row["view"] != view:
            continue
        if row["video_sha256"] != manifest["video_sha256"] or any(
            row["segment"][k] != manifest["segment"][k]
            for k in ("start_frame", "end_frame", "fps")
        ):
            raise ValueError("Manual observation metadata source mismatch")
        for note in result:
            if note["item"] in row.get("criterion_holds", []):
                note["eligible"] = False
                note["holds"].append("manual_input_audit_peak_outside_image")
    return result
