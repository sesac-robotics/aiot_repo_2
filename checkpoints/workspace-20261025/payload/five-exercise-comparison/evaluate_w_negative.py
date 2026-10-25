"""이미 노출된 W 미수행 학습 자료에서 실패 검출 가능성을 검사한다."""

import numpy as np

import audit
from extract import extract_view
import w_morphology
import ytw_phases


def run() -> dict:
    """기존 W 0등급 초안의 자동 단계·모양 결과를 별도로 보존한다.

    Returns:
        dict: 자동 등급 또는 관측·단계 실패 사유. 독립 시험이 아니다.

    Raises:
        ValueError: 원본·캐시·입력 구간 계약 오류.
        RuntimeError: 자원 보호·관절 추출 실패.
        OSError: 원본 읽기·진단 결과 저장 실패.
    """
    row = audit.read_json(audit.OUT / "w_prior_negative_review.json")
    extracted = extract_view(
        row, "primary", row["video_path"], audit.OUT / "w_prior_negative_cache"
    )
    with np.load(extracted["cache"]) as data:
        sequence = data["image_xy"]
    phases = ytw_phases.detect(sequence)
    window = phases["phases"].get("W_attempt")
    result = (
        w_morphology.evaluate(
            sequence, window["start_index"], window["end_index"]
        )
        if window
        else dict(grade=None, hold_reason="W_attempt_not_detected")
    )
    report = dict(
        id=row["id"],
        draft_grade=0,
        automatic=result,
        phase_detection=phases,
        extraction=extracted,
        independent_final_test=False,
        exposed_training_negative=True,
    )
    audit.save_json(audit.OUT / "w_prior_negative_results.json", report)
    return report


if __name__ == "__main__":
    print(run())
