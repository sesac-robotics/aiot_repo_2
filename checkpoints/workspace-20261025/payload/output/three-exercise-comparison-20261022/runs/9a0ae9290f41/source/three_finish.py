"""현재 후보의 시간·관절 영상·관측 범위·보존 검증을 순서대로 완료한다.

전체 실행은 기존 단일 작업 잠금·온도·전원·메모리 보호 안에서 수행한다.
각 단계는 독립 JSON을 저장하며 수치 미달을 완료로 바꾸지 않는다.
"""

import audit
import three_protocol
import three_publish
import three_quality_summary
import three_runtime
import three_score_timing
from resources import check_resources


def run() -> dict:
    """3종 현재 후보의 개발 검증 산출물을 단계별로 저장한다.

    Returns:
        dict: 단계 완료·최종 HTML·완료 기준 미충족을 포함한 실행 결과.

    Raises:
        OSError: 원본·후보·결과 파일 접근 실패.
        ValueError: 예측·관측·동기화·보존 계약 불일치.
        RuntimeError: 자원 보호 또는 하위 검증 실행 실패.
    """
    root = three_protocol.OUT
    state = dict(completed_stages=[], all_three_accepted=False)
    for name, function in [
        ("runtime", three_runtime.run),
        ("score_timing", three_score_timing.run),
        ("overlays", three_publish.overlays),
        ("quality_summary", three_quality_summary.run),
    ]:
        check_resources()
        function()
        state["completed_stages"].append(name)
        audit.save_json(root / "finalization_checkpoint.json", state)
    state["html"] = str(three_publish.publish())
    state["completed_stages"].append("publish")
    state["completed"] = True
    audit.save_json(root / "finalization_checkpoint.json", state)
    return state


if __name__ == "__main__":
    print(run())
