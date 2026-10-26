"""관절 캐시를 재사용하여 W 개발 진단을 순서대로 갱신한다."""

import evaluate_w_challenge
import evaluate_w_challenge_v2
import evaluate_w_negative


def run() -> None:
    """두 개발 후보 묶음과 기존 음성 사례를 순서대로 평가한다.

    Returns:
        None: 각 모듈의 JSON을 저장한다. 독립 최종 시험이 아니다.

    Raises:
        ValueError: 고정 규칙·캐시·구간 계약 불일치.
        OSError: 입력 읽기·결과 저장 실패.
        RuntimeError: 자원 보호 조건 위반.
    """
    print(evaluate_w_challenge.run()["summary"])
    print(evaluate_w_challenge_v2.run()["summary"])
    print(evaluate_w_negative.run()["automatic"])


if __name__ == "__main__":
    run()
