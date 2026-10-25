"""팀 교정 엔진을 유지하며 같은 카메라의 자세 점수를 별도 프로세스로 연결합니다."""

import ast
import json
import threading
import time
import uuid
from pathlib import Path

from Server.ExerciseEvaluation.contracts import CORRECTION_CODES, validate
from Server.ExerciseEvaluation.coordinator import Coordinator, EvaluationWorker
from Server.ExerciseEvaluation.preview_privacy import mosaic_faces
from Server.ExerciseEvaluation.score_recording import write_json

try:
    from . import direct_correction_server as recording
except ImportError:
    import direct_correction_server as recording


def original_recording_engine(source_root, sessions_root, reader_factory=None):
    """원본 루프와 저장 사본이 저장 호출 외에 동일한지 확인합니다."""
    source_root = Path(source_root)
    original_tree = ast.parse((source_root / "live_app/engine.py").read_text())
    copied_tree = ast.parse(Path(recording.__file__).read_text())

    def method(tree, name):
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == name
        )
        return next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "run"
        )

    class Strip(ast.NodeTransformer):
        def visit_Expr(self, node):
            call = node.value
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr
                in {"begin_recording", "capture_frame", "finish_recording"}
            ):
                return None
            return node

    if ast.dump(method(original_tree, "LiveEngine")) != ast.dump(
        Strip().visit(method(copied_tree, "RecordingMixin"))
    ):
        raise RuntimeError("팀 교정 원본과 저장 루프가 달라 통합을 중단합니다.")
    import live_app.engine as original

    if (
        Path(original.__file__).resolve()
        != (source_root / "live_app/engine.py").resolve()
    ):
        raise RuntimeError("교정 PYTHONPATH와 선택한 원본 경로가 다릅니다.")

    # Configure only this process's original reader; source files stay intact.
    original.cv2 = WebcamVgaCapture(original.cv2)

    recording.__dict__.update(
        cv2=original.cv2,
        np=original.np,
        TemporalGate=original.TemporalGate,
        ROOT=original.ROOT,
    )
    base_reader = reader_factory or original.CameraReader
    original.CameraReader = lambda source: PausingReader(
        base_reader(source), source
    )

    class Engine(recording.RecordingMixin, original.LiveEngine):
        integration_session_id = None

        def begin_recording(self, exercise):
            super().begin_recording(exercise)
            if self.writer:
                self.writer.metadata["integration_session_id"] = (
                    self.integration_session_id
                )
                write_json(
                    self.writer.directory / "metadata.json",
                    self.writer.metadata,
                )

        def capture_frame(
            self, raw, result, stamp, sequence, epoch, width, height
        ):
            super().capture_frame(
                raw, result, stamp, sequence, epoch, width, height
            )
            # The original loop calls this while holding its publication lock.
            # Its skeleton rendering is retained; only display privacy is added.
            image = original.cv2.imdecode(
                original.np.frombuffer(self.jpeg, original.np.uint8),
                original.cv2.IMREAD_COLOR,
            )
            xy = original.np.full((33, 2), original.np.nan)
            if raw is not None:
                xy = raw[:, :2] * [image.shape[1], image.shape[0]]
                xy[raw[:, 3] < 0.5] = original.np.nan
            masked = mosaic_faces(image, xy)
            if not masked:
                image = original.cv2.resize(
                    original.cv2.resize(image, (16, 12)),
                    (image.shape[1], image.shape[0]),
                    interpolation=original.cv2.INTER_NEAREST,
                )
            self.jpeg = original.cv2.imencode(".jpg", image)[1].tobytes()

    return Engine(sessions_root, source_root)


class WebcamVgaCapture:
    """웹캠 장치를 열 때 가로 VGA를 요청하고 다른 영상 입력은 유지합니다."""

    def __init__(self, backend):
        self.backend = backend

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def VideoCapture(self, source, *args, **kwargs):
        capture = self.backend.VideoCapture(source, *args, **kwargs)
        if isinstance(source, int) and capture.isOpened():
            capture.set(self.backend.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(self.backend.CAP_PROP_FRAME_HEIGHT, 480)
        return capture


class PausingReader:
    """실제 카메라 하나를 유지하면서 교정의 입력만 일시 정지합니다."""

    def __init__(self, reader, source=None):
        self.reader = reader
        self.paused = False
        self.rotation = 0
        self.cached_key = None
        self.cached_item = None
        if source and Path(str(source)).is_file():
            import cv2

            probe = cv2.VideoCapture(str(source))
            if probe.isOpened() and not probe.get(
                cv2.CAP_PROP_ORIENTATION_AUTO
            ):
                self.rotation = (
                    int(probe.get(cv2.CAP_PROP_ORIENTATION_META)) % 360
                )
            probe.release()

    def start(self):
        self.reader.start()

    def stop(self):
        self.reader.stop()

    @property
    def transport(self):
        return self.reader.transport

    def raw_snapshot(self):
        seq, epoch, item, connected, error, fps = self.reader.snapshot()
        if item is not None and self.rotation:
            import cv2

            key = epoch, seq
            if key != self.cached_key:
                frame, stamp = item
                code = {
                    90: cv2.ROTATE_90_CLOCKWISE,
                    180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
                }.get(self.rotation)
                self.cached_item = (
                    (cv2.rotate(frame, code), stamp)
                    if code is not None
                    else item
                )
                self.cached_key = key
            item = self.cached_item
        return seq, epoch, item, connected, error, fps

    def snapshot(self):
        seq, epoch, item, connected, error, fps = self.raw_snapshot()
        if self.paused:
            return seq, epoch, None, False, "일시정지", fps
        return seq, epoch, item, connected, error, fps


class SharedScoreCamera:
    """교정이 소유한 원본 프레임을 읽고 장치를 추가로 열지 않습니다."""

    selected = "shared_camera"

    def __init__(self, bridge):
        self.bridge = bridge
        self.origin = None
        self.last_tick = -1

    def start(self):
        pass

    def stop(self):
        pass

    def snapshot(self):
        reader = self.bridge.engine.reader
        if reader is None or not self.bridge.enabled:
            return 0, 0, None, False, "교정 카메라 준비 중", 0
        # Score reads the same unannotated capture independently at its own rate.
        seq, epoch, item, connected, error, fps = reader.raw_snapshot()
        if self.bridge.paused:
            return seq, epoch, None, connected, "", fps
        if item is not None:
            import cv2

            frame, stamp = item
            if self.origin is None:
                self.origin = stamp
            tick = round((stamp - self.origin) * 10)
            if tick <= self.last_tick:
                return seq, epoch, None, connected, error, fps
            self.last_tick = tick
            h, w = frame.shape[:2]
            scale = min(1.0, 640 / w, 480 / h)
            if scale < 1:
                frame = cv2.resize(frame, (round(w * scale), round(h * scale)))
            item = frame, stamp
        return seq, epoch, item, connected, error, fps


class IntegratedSession:
    fixed_mode = "combined"

    def __init__(self, engine, score_python, sessions_root, replay_source=None):
        self.engine = engine
        self.sessions_root = Path(sessions_root)
        self.replay_source = replay_source
        self.lock = threading.RLock()
        self.control = threading.RLock()
        self.enabled = self.paused = False
        self.session_id = None
        self.started = None
        self.events = []
        self.last_summary = None

        def worker_factory(*args):
            worker = EvaluationWorker(*args)
            worker.provisional_scores = False
            return worker

        self.score = Coordinator(
            {"score": str(score_python)},
            fixed_mode="score",
            camera_factory=lambda _: SharedScoreCamera(self),
            worker_factory=worker_factory,
        )

    @property
    def jpeg(self):
        return self.engine.image() if self.session_id else None

    def snapshot(self):
        state = self.score.snapshot()
        correction = self.engine.snapshot()
        if self.paused:
            state["score"] = dict(
                self.score.state.get("score", {}),
                live_score_100=None,
                paused=True,
            )
            correction = dict(correction, result=None, status="paused")
        return dict(
            state,
            session_id=self.session_id,
            mode="combined",
            running=state.get("running", False),
            paused=self.paused,
            status="paused" if self.paused else state.get("status"),
            correction=correction,
            integrated_summary=self.last_summary,
            recording_scope="동일 카메라의 교정·점수 입력을 각각 저장하며 공통 세션으로 연결합니다.",
        )

    def start(self, options):
        if options.get("mode", "combined") != "combined":
            raise ValueError("이 서버는 운동 교정·자세 점수 통합 전용입니다.")
        checked = validate(dict(options, mode="score"), "score")
        if checked["camera"] == "auto":
            raise ValueError("통합 모드는 webcam 또는 esp32를 명시하십시오.")
        with self.control:
            self.stop()
            self.session_id = uuid.uuid4().hex
            self.started = time.time()
            self.events = [
                {
                    "event": "start",
                    "monotonic": time.monotonic(),
                    "unix": self.started,
                }
            ]
            self.last_summary = None
            self.engine.integration_session_id = self.session_id
            camera = checked["camera"]
            source = self.replay_source or (
                checked["esp32"]
                if camera == "esp32"
                else str(checked["webcam"])
            )
            self.options = dict(checked, mode="combined")
            try:
                self.engine.start(
                    source,
                    CORRECTION_CODES.get(
                        checked["exercise"], checked["exercise"]
                    ),
                    True,
                )
                self.score.start(checked)
                with self.score.lock:
                    self.score.state["session_id"] = self.session_id
                self.enabled = True
            except Exception:
                self.stop()
                raise

    def pause(self):
        with self.control:
            if not self.score.state.get("running"):
                raise ValueError("실행 중인 운동이 없습니다.")
            self.paused = True
            self.events.append(
                {"event": "pause", "monotonic": time.monotonic()}
            )
            if self.engine.reader:
                self.engine.reader.paused = True
            worker = self.score.workers.get("score")
            if worker:
                worker.reset_motion()

    def resume(self):
        with self.control:
            if not self.paused:
                return
            worker = self.score.workers.get("score")
            if worker:
                worker.reset_motion()
            self.engine.reset()
            if self.engine.reader:
                self.engine.reader.paused = False
            self.paused = False
            self.events.append(
                {"event": "resume", "monotonic": time.monotonic()}
            )

    def stop(self):
        with self.control:
            self.score.stop()
            self.engine.stop()
            self.enabled = self.paused = False
            if not self.session_id or self.last_summary:
                return
            state = self.snapshot()
            correction = state["correction"]
            score = state.get("score", {})
            summary = dict(
                schema="integrated-session-v1",
                control_events=self.events
                + [{"event": "stop", "monotonic": time.monotonic()}],
                session_id=self.session_id,
                started_unix=self.started,
                ended_unix=time.time(),
                options=self.options,
                score=score,
                correction=correction,
                correction_directory=correction.get("session_path"),
                score_directory=score.get("recording", {}).get("directory"),
                temporal_contract="교정은 연속 시간창 결과이며 특정 반복의 확정 오류 라벨로 환산하지 않습니다.",
                recording_errors=[
                    e
                    for e in (
                        correction.get("recording_error"),
                        score.get("recording", {}).get("error"),
                    )
                    if e
                ],
            )
            score_directory = summary.get("score_directory")
            summary["score_epoch_directories"] = (
                [
                    str(p.parent)
                    for p in sorted(
                        Path(score_directory).parent.glob(
                            "epoch-*/metadata.json"
                        )
                    )
                ]
                if score_directory
                else []
            )
            self.sessions_root.joinpath(self.session_id).mkdir(
                parents=True, exist_ok=True
            )
            write_json(
                self.sessions_root / self.session_id / "summary.json", summary
            )
            self.last_summary = summary
