"""공유 파이프라인 계측: 단조 시계, 짧은 잠금, 제한된 표본 보관."""

import math
import threading
import time
from collections import deque


class PipelineMetrics:
    def __init__(
        self, window_seconds=5.0, sample_size=300, clock=time.monotonic
    ):
        if (
            not math.isfinite(window_seconds)
            or window_seconds <= 0
            or sample_size < 1
        ):
            raise ValueError("측정 구간과 표본 수는 양수여야 합니다.")
        self.window_seconds = window_seconds
        self.sample_size = sample_size
        self._clock = clock
        self._lock = threading.Lock()
        self._started = clock()
        self._events = {
            name: deque() for name in ("capture", "inference", "processed")
        }
        self._totals = dict.fromkeys(self._events, 0)
        self._times = {
            name: deque(maxlen=sample_size)
            for name in ("inference", "mediapipe_inference", "jpeg_encoding")
        }
        self._latest_capture = None
        self._skipped = 0

    def _prune(self, events, now):
        # 요청 빈도와 무관하게 최근 시간 구간의 이벤트만 보관합니다.
        cutoff = now - self.window_seconds
        while events and events[0] <= cutoff:
            events.popleft()

    def record_event(self, name, skipped=0):
        with self._lock:
            now = self._clock()
            events = self._events[name]
            self._prune(events, now)
            events.append(now)
            self._totals[name] += 1
            if name == "capture":
                self._latest_capture = now
                self._skipped += skipped

    def record_duration(self, name, milliseconds):
        # 프레임 경로에서는 정렬/로그 없이 표본 하나만 추가합니다.
        with self._lock:
            self._times[name].append(milliseconds)

    def snapshot(self):
        with self._lock:
            now = self._clock()
            elapsed = min(self.window_seconds, max(0.0, now - self._started))
            for events in self._events.values():
                self._prune(events, now)
            rates = {
                name: len(events) / elapsed if elapsed else 0.0
                for name, events in self._events.items()
            }
            samples = {
                name: list(values) for name, values in self._times.items()
            }
            totals = dict(self._totals)
            age = (
                None
                if self._latest_capture is None
                else (now - self._latest_capture) * 1000
            )
            skipped = self._skipped

        # P95 정렬은 수신/추론 잠금 밖에서 API 조회 또는 로그 시에만 수행합니다.
        timings = {}
        for name, values in samples.items():
            ordered = sorted(values)
            timings[name + "_ms"] = {
                "latest": values[-1] if values else None,
                "mean": sum(values) / len(values) if values else None,
                "p95": (
                    ordered[math.ceil(len(ordered) * 0.95) - 1]
                    if ordered
                    else None
                ),
                "samples": len(values),
            }
        return {
            "fps_window_seconds": self.window_seconds,
            "timing_sample_limit": self.sample_size,
            "camera_capture_fps": rates["capture"],
            "pose_inference_fps": rates["inference"],
            "processing_fps": rates["processed"],
            "latest_frame_age_ms": age,
            "skipped_frames": skipped,
            "captured_frames": totals["capture"],
            "inferred_frames": totals["inference"],
            "processed_frames": totals["processed"],
            **timings,
        }
