"""검출과 점수에 같은 영상 경계 2% 관측 계약을 적용한다.

원시 관절은 저장용으로 보존하며, 머리·발목 누락을 전신 보류로 확대하지 않는다.
"""
import numpy as np

VERSION = 'live-observation-border2-v1'
MINIMUM_COVERAGE = .8
JOINT_NAMES = {11:'왼쪽 어깨',12:'오른쪽 어깨',13:'왼쪽 팔꿈치',
               14:'오른쪽 팔꿈치',15:'왼쪽 손목',16:'오른쪽 손목',
               23:'왼쪽 골반',24:'오른쪽 골반',25:'왼쪽 무릎',
               26:'오른쪽 무릎',27:'왼쪽 발목',28:'오른쪽 발목'}


def mask_frame_border(points, width, height):
    """N×33×2 또는 33×2 사본에 배포 점수기와 같은 경계를 적용한다."""
    x = np.asarray(points, dtype=float).copy()
    if x.shape[-2:] != (33, 2) or width <= 0 or height <= 0:
        raise ValueError('Valid pixel joints and image dimensions required')
    low = np.array([width, height]) * .02
    high = np.array([width, height]) * .98
    invalid = ~np.isfinite(x).all(axis=-1) | ((x < low) | (x > high)).any(axis=-1)
    x[invalid] = np.nan
    return x


def movement_observation(points, width, height, exercise, view):
    """운동 검출에 필요한 관절만 검사하고 구도 안내를 만든다."""
    raw = np.asarray(points, dtype=float)
    x = mask_frame_border(raw, width, height)
    required = [11, 12, 23, 24]
    if exercise == 'squats' and view != 'front' or exercise == 'plank':
        required += [25, 26, 27, 28]
    elif exercise in ('shoulder', 'pushups', 'ytw'):
        required += [13, 14, 15, 16]
    missing = [j for j in required if not np.isfinite(x[j]).all()]
    if exercise == 'squats' and view == 'front' and not np.isfinite(x[[25,26]]).all(axis=1).any():
        missing += [25, 26]
    edge = [j for j in missing if np.isfinite(raw[j]).all()]
    names = [JOINT_NAMES[j] for j in missing]
    message = ('관측 준비 완료 · 필요한 관절이 화면 안쪽에 보입니다.' if not missing else
               '관측 대기 · ' + '·'.join(names) + '이 화면 안쪽에 보이도록 구도를 맞춰 주세요.')
    if edge:
        message += ' 화면 경계 또는 밖의 추정 좌표는 횟수와 점수에서 함께 제외합니다.'
    if exercise == 'ytw':
        message = 'YTW 자동 반복 구분은 아직 미지원입니다. 관절 저장과 완료 회차 평가는 다릅니다.'
    return dict(version=VERSION,ready=not missing,missing=names,
                edge_joints=[JOINT_NAMES[j] for j in edge],message=message)


def segment_coverage(samples):
    """실제 10Hz 시간축의 관측률. 검출기가 건너뛴 슬롯을 분모에 남긴다."""
    if not samples:
        return 0.
    return len({tick for tick, _ in samples}) / (samples[-1][0] - samples[0][0] + 1)
