"""하체가 없어도 직접 보이는 팔 지지 위치 변경만 검출하는 실험 규칙이다.

변경 검출은 지지 지속성 2등급을 배제하는 근거다. 중단 0등급 여부는 별도 확인한다. 변화가 없다는 이유로
하체 지지·정상 지속성 2등급 또는 총점 만점을 부여하지 않는다.
"""

import numpy as np


def detect(sequence: np.ndarray) -> dict:
    """몸통 길이로 정규화한 어깨 대비 팔꿈치의 처음·끝 변위를 구한다.

    Args:
        sequence: N×33×2 픽셀 좌표. 미관측은 NaN이다.

    Returns:
        dict: 선택 측면, 동시 관측률, 몸통 길이 비율 변위, 지속성 등급 상한.
        80% 이상 관측·5표본 이상에서 변위 0.20 이상이면 지속성 등급 상한을 1로 제한한다.
        그 외 continuity_grade_upper_bound는 None이다. 이는 정상·실제 접촉 보증이 아니다.
        고정 구도에서 쓰며 카메라 회전·몸통 회전·추적 오차에 민감하다.

    Raises:
        ValueError: 좌표 형상 오류 또는 빈 입력.
    """
    seq = np.asarray(sequence, dtype=float)
    if seq.ndim != 3 or seq.shape[1:] != (33, 2) or not len(seq):
        raise ValueError("Expected nonempty N x 33 x 2 pixels")
    options = []
    for side, shoulder, elbow, hip in [
        ("left", 11, 13, 23),
        ("right", 12, 14, 24),
    ]:
        valid = np.isfinite(seq[:, [shoulder, elbow, hip]]).all((1, 2))
        torso = np.linalg.norm(seq[:, shoulder] - seq[:, hip], axis=1)
        valid &= torso > 1
        relative = np.full((len(seq), 2), np.nan)
        relative[valid] = (seq[valid, elbow] - seq[valid, shoulder]) / torso[
            valid, None
        ]
        options.append((float(valid.mean()), side, relative))
    coverage, side, relative = max(options, key=lambda x: x[0])
    window = max(2, len(seq) // 5)
    first = relative[:window]
    last = relative[-window:]
    first = first[np.isfinite(first).all(1)]
    last = last[np.isfinite(last).all(1)]
    measured = None
    if len(seq) >= 5 and coverage >= 0.8 and len(first) >= 2 and len(last) >= 2:
        measured = float(
            np.linalg.norm(np.median(last, axis=0) - np.median(first, axis=0))
        )
    return dict(
        side=side,
        coverage=coverage,
        relative_displacement=measured,
        threshold=0.20,
        continuity_grade_upper_bound=(
            1 if measured is not None and measured >= 0.20 else None
        ),
        total_score=None,
        absence_does_not_mean_normal=True,
        expert_validated=False,
    )
