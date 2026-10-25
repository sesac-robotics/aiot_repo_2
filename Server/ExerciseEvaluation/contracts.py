"""운동 코드·실행 모드·세션 경계를 외부 모델 의존성 없이 정의합니다."""

EXERCISES = {
    "squats": "스쿼트",
    "pushups": "푸시업",
    "shoulder": "숄더 프레스",
    "plank": "엘보 플랭크",
    "ytw": "YTW",
}
CORRECTION_CODES = {"shoulder": "shoulder_press", "plank": "elbow_plank"}
MODES = {
    "combined": ("correction", "score"),
    "correction": ("correction",),
    "score": ("score",),
}


class ObservedCadence:
    """실제 프레임을 시간 슬롯마다 하나만 선택하고 누락 슬롯을 알립니다."""

    def __init__(self, fps):
        self.fps = fps
        self.origin = None
        self.last_slot = None

    def observe(self, timestamp):
        """중복 슬롯은 None, 새 슬롯은 연속 여부를 반환하며 보간하지 않습니다."""
        if self.origin is None:
            self.origin = timestamp
        slot = int((timestamp - self.origin) * self.fps + 1e-8)
        if self.last_slot is not None and slot <= self.last_slot:
            return None
        continuous = self.last_slot is None or slot == self.last_slot + 1
        self.last_slot = slot
        return continuous


def validate(options, fixed_mode=None):
    """지원 옵션만 허용하고 동시 실행 전용 진입점의 모드 변경을 차단합니다."""
    result = dict(options)
    result.setdefault("exercise", "squats")
    result.setdefault("mode", fixed_mode or "combined")
    result.setdefault("camera", "webcam")
    result.setdefault("esp32", "http://192.168.15.208:81/stream")
    result.setdefault("webcam", "0")
    result.setdefault("view", "front")
    if result["exercise"] not in EXERCISES or result["mode"] not in MODES:
        raise ValueError("지원하지 않는 운동 또는 실행 모드입니다.")
    if fixed_mode and result["mode"] != fixed_mode:
        raise ValueError("이 실행 파일은 교정·점수 동시 실행 전용입니다.")
    if result["camera"] not in ("esp32", "webcam", "auto"):
        raise ValueError("카메라는 esp32, webcam, auto 중 선택합니다.")
    from urllib.parse import urlsplit

    if urlsplit(str(result["esp32"])).scheme not in ("http", "https"):
        raise ValueError("ESP32는 HTTP(S) 주소를 입력합니다.")
    if not str(result["webcam"]).isdigit():
        raise ValueError("웹캠은 0 이상의 장치 번호를 입력합니다.")
    if result["view"] != "front":
        raise ValueError("통합 실행은 현재 정면 구도만 지원합니다.")
    return result
