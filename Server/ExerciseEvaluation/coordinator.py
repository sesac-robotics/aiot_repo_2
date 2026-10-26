"""카메라 한 대의 실제 관측을 독립된 교정·점수 프로세스에 전달합니다."""

import base64
import copy
import json
import queue
import struct
import subprocess
import threading
import time
import uuid

from .contracts import MODES, validate
from .paths import REPO, worker_environment


class EvaluationWorker:
    """크기 1 입력 큐로 추론 지연 누적을 막고 실제 프레임 시간을 보존합니다."""

    def __init__(self, kind, exercise, python, callback):
        self.kind = kind
        self.callback = callback
        self.queue = queue.Queue(maxsize=1)
        self.stopped = threading.Event()
        self.dropped = 0
        self.last_meta = None
        self.closing = False
        self.process = subprocess.Popen(
            [python, "-m", "Server.ExerciseEvaluation.worker", kind, exercise],
            cwd=REPO,
            env=worker_environment(kind),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
        )
        self.threads = [
            threading.Thread(target=self._send, daemon=True),
            threading.Thread(target=self._receive, daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def submit(self, frame, meta):
        """복제·보간 없이 가장 최근 실제 프레임만 대기시킵니다."""
        item = (frame, {**meta, "shape": list(frame.shape)})
        self.last_meta = meta
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(item)
            except queue.Full:
                pass

    def _send(self):
        try:
            while not self.stopped.is_set():
                try:
                    frame, meta = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if hasattr(self, "provisional_scores"):
                    meta = dict(
                        meta, provisional_scores=self.provisional_scores
                    )
                packet = (
                    json.dumps(meta).encode()
                    + b"\n"
                    + (frame.tobytes() if frame is not None else b"")
                )
                self.process.stdin.write(
                    struct.pack("!I", len(packet)) + packet
                )
                self.process.stdin.flush()
                if meta.get("command") == "stop":
                    return
        except (BrokenPipeError, OSError, ValueError):
            pass

    def _receive(self):
        try:
            for line in self.process.stdout:
                result = json.loads(line)
                result["dropped_input_frames"] = self.dropped
                self.callback(self.kind, result)
        except (ValueError, OSError):
            pass
        finally:
            if (
                not self.stopped.is_set()
                and not self.closing
                and self.last_meta
            ):
                self.callback(
                    self.kind,
                    {
                        **self.last_meta,
                        "status": "unavailable",
                        "error": "평가 프로세스가 종료됐습니다. 실행 환경과 터미널 로그를 확인하십시오.",
                    },
                )

    def reset_motion(self):
        """입력 순서를 유지한 제어 패킷으로 진행 중 동작만 초기화합니다."""
        if self.last_meta and self.process.poll() is None:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(
                (
                    None,
                    {
                        **self.last_meta,
                        "timestamp": time.monotonic(),
                        "command": "reset_motion",
                    },
                )
            )

    def close(self):
        """처리 중인 프레임과 무관하게 제한 시간 안에 프로세스를 종료합니다."""
        self.closing = True
        if self.last_meta and self.process.poll() is None:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait((None, {**self.last_meta, "command": "stop"}))
        elif self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        self.stopped.set()
        for thread in self.threads:
            thread.join(timeout=1)
        self.process.stdin.close()
        self.process.stdout.close()


class Coordinator:
    """선택 모드의 평가만 시작하고 세션·소스가 다른 늦은 결과를 버립니다."""

    def __init__(
        self,
        runtimes,
        fixed_mode=None,
        camera_factory=None,
        worker_factory=EvaluationWorker,
    ):
        self.runtimes = runtimes
        self.fixed_mode = fixed_mode
        self.camera_factory = camera_factory
        self.worker_factory = worker_factory
        self.lock = threading.RLock()
        self.control = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.camera = None
        self.workers = {}
        self.generation = 0
        self.jpeg = None
        self.state = {"running": False, "status": "idle"}

    def snapshot(self):
        """현재 프레임과 결과를 분리한 복사본을 반환합니다."""
        with self.lock:
            state = copy.deepcopy(self.state)
            now = time.monotonic()
            for kind in ("correction", "score"):
                result = state.get(kind, {})
                if (
                    result.get("timestamp")
                    and result.get("status") != "stopped"
                    and now - result["timestamp"] > 1
                ):
                    state[kind] = {
                        "status": "stale",
                        "message": "최신 관측을 기다립니다.",
                    }
            return state

    def bind_session(self, session_id: str) -> None:
        """공유 카메라 입력을 열기 전에 공통 세션 ID를 연결합니다."""
        if not session_id:
            raise ValueError("세션 ID가 필요합니다.")
        with self.lock:
            self.state["session_id"] = session_id

    def retained_score(self) -> dict:
        """일시정지 표시용 마지막 결과를 잠금 안에서 복사합니다."""
        with self.lock:
            return copy.deepcopy(self.state.get("score", {}))

    def reset_motion(self) -> None:
        """점수 Worker의 진행 중 동작만 초기화하며 완료 결과는 유지합니다."""
        with self.lock:
            worker = self.workers.get("score")
            if worker:
                worker.reset_motion()

    def receive(self, kind, result):
        """현재 세션·연결 구간에서 나온 최신 결과만 게시합니다."""
        with self.lock:
            if (
                (
                    (
                        self.state.get("running")
                        and self.state.get("status") == "running"
                    )
                    or result.get("final") is True
                )
                and result.get("generation") == self.generation
                and result.get("epoch") == self.state.get("epoch")
            ):
                previous = self.state.get(kind, {}).get("timestamp", -1)
                if result.get("timestamp", -1) >= previous:
                    result = dict(result)
                    preview = result.pop("preview_jpeg_b64", None)
                    if kind == "score" and preview is not None:
                        try:
                            self.jpeg = base64.b64decode(preview, validate=True)
                            self.state["face_mosaic"] = result.get(
                                "face_mosaic"
                            )
                        except (ValueError, TypeError):
                            self.jpeg = None
                    self.state[kind] = result

    def start(self, options):
        """시작 요청에서만 장치를 열며 새 세션에 이전 측정을 섞지 않습니다."""
        options = validate(options, self.fixed_mode)
        with self.control:
            self._stop()
            if self.camera_factory is None:
                from .camera_sources import ChoiceCamera

                factory = ChoiceCamera
            else:
                factory = self.camera_factory
            with self.lock:
                self.generation += 1
                self.state = {
                    "running": True,
                    "status": "connecting",
                    "session_id": uuid.uuid4().hex,
                    "generation": self.generation,
                    "epoch": None,
                    "options": options,
                    "correction": {"status": "not_requested"},
                    "score": {"status": "not_requested"},
                }
                self.jpeg = None
            self.stop_event = threading.Event()
            try:
                for kind in MODES[options["mode"]]:
                    try:
                        self.workers[kind] = self.worker_factory(
                            kind,
                            options["exercise"],
                            self.runtimes[kind],
                            self.receive,
                        )
                        self.state[kind] = {"status": "loading"}
                    except Exception as error:
                        self.state[kind] = {
                            "status": "unavailable",
                            "error": str(error),
                        }
                self.camera = factory(options)
                self.camera.start()
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
            except Exception:
                self._stop()
                raise

    def _run(self):
        seen = None
        last_source = None
        last_shape = None
        effective_epoch = 0
        try:
            while not self.stop_event.is_set():
                seq, epoch, item, connected, error, fps = self.camera.snapshot()
                shape = tuple(item[0].shape) if item is not None else last_shape
                if epoch != last_source or shape != last_shape:
                    effective_epoch += 1
                    last_source, last_shape = epoch, shape
                epoch = effective_epoch
                with self.lock:
                    if epoch != self.state.get("epoch"):
                        self.state.update(
                            epoch=epoch,
                            continuity_notice="연결 구간 변경 · 진행 중 동작과 반복 목록을 초기화합니다.",
                        )
                        for kind in self.workers:
                            self.state[kind] = {"status": "warming_up"}
                    self.state.update(
                        status="running" if connected else "connecting",
                        camera=self.camera.selected,
                        capture_fps=fps,
                        camera_error=error,
                    )
                    if not connected:
                        self.jpeg = None
                        for kind in self.workers:
                            self.state[kind] = {"status": "camera_unavailable"}
                if connected and item is not None and seen != (epoch, seq):
                    frame, stamp = item
                    if time.monotonic() - stamp <= 0.35:
                        seen = (epoch, seq)
                        meta = {
                            "session_id": self.state["session_id"],
                            "generation": self.generation,
                            "epoch": epoch,
                            "frame_id": seq,
                            "timestamp": stamp,
                        }
                        for worker in self.workers.values():
                            worker.submit(frame, meta)
                        # 미리보기는 점수 작업자가 같은 프레임의 관절로 모자이크한 것만 게시합니다.
                    else:
                        with self.lock:
                            self.jpeg = None
                            self.state["status"] = "connecting"
                self.stop_event.wait(0.01)
        except Exception as error:
            with self.lock:
                self.state.update(status="failed", camera_error=str(error))

    def stop(self):
        """촬영과 선택된 평가를 종료합니다."""
        with self.control:
            self._stop()

    def _stop(self):
        with self.lock:
            self.state["running"] = False
            self.state["status"] = "idle"
            self.jpeg = None
        self.stop_event.set()
        if self.camera:
            self.camera.stop()
        if self.thread:
            self.thread.join(timeout=15)
            if self.thread.is_alive():
                raise RuntimeError(
                    "카메라 종료 대기 중입니다. 다시 시작하지 않습니다."
                )
        for worker in self.workers.values():
            worker.close()
        self.workers = {}
        self.camera = None
        self.thread = None
        with self.lock:
            self.state.update(running=False, status="idle")
