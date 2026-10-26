"""스쿼트 모델의 CPU 실행을 합성 입력으로 검사한다.

입력은 고정 난수의 무단위 텐서이며 실제 운동 영상이 아니다.
출력 JSON은 모델 구조와 실행 가능성만 기록하며 자세 점수를 제공하지 않는다.
체크포인트 누락, 구조 불일치, 비정상 출력이면 예외로 실패한다.
"""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [
    str(ROOT / "support-dependencies"),
    str(ROOT / "runtime/transformer_rehabilitation"),
]

import torch
from models import TemporalModel


def main():
    """합성 입력의 추론을 실행하고 검사 JSON을 저장한다.

    Args:
        없음. 체크포인트 경로는 이 파일 위치를 기준으로 결정한다.

    Returns:
        None. 출력 형태, 유한성, 실행 시간(초)을 파일과 표준 출력에 기록한다.

    Raises:
        RuntimeError: 가중치가 구조와 다르거나 출력 형태ㆍ값이 잘못된 경우.
        FileNotFoundError: 학습된 가중치 파일이 없는 경우.
    """
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = TemporalModel().eval()
    weights = torch.load(
        ROOT / "checkpoints/Kimore_ex5.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(weights, strict=True)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(torch.randn(1, 100, 25, 3))
    if output.shape != (1, 1) or not torch.isfinite(output).all():
        raise RuntimeError("출력 형태 또는 유한성 검사 실패")
    result = {
        "status": "passed",
        "device": "cpu",
        "input_shape": [1, 100, 25, 3],
        "output_shape": list(output.shape),
        "elapsed_seconds": time.perf_counter() - started,
        "purpose": "합성 입력의 실행 검사. 실제 자세 점수ㆍ정확도 검증 아님.",
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    (ROOT / "squat-runtime-result.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
