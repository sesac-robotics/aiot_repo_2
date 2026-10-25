"""팀 교정 루프의 별도 사본에 입력 저장을 추가한 로컬 서버입니다.

run()은 팀 원본에서 복사했고 begin/capture/finish 저장 호출만 추가했습니다.
원본 파일·모델·판정 기준은 변경하지 않습니다.
"""

import argparse
import ast
import copy
import json
import signal
import threading
import time
from collections import deque
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from .correction_session_store import SessionReader, SessionWriter, digest
except ImportError:
    from correction_session_store import SessionReader, SessionWriter, digest

# Arguments are resolved before importing the team's original modules.
parser = argparse.ArgumentParser()
parser.add_argument("--correction-root", type=Path, required=True)
parser.add_argument("--sessions", type=Path, required=True)
parser.add_argument("--port", type=int, default=8782)


def serve(args):
    # A new upstream algorithm must be reviewed before using this copied loop.
    source_tree = ast.parse(
        (args.correction_root / "live_app/engine.py").read_text()
    )
    copied_tree = ast.parse(Path(__file__).read_text())

    def find_run(tree, class_name):
        return next(
            f
            for c in tree.body
            if isinstance(c, ast.ClassDef) and c.name == class_name
            for f in c.body
            if isinstance(f, ast.FunctionDef) and f.name == "run"
        )

    class StripRecording(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr
                in ("begin_recording", "capture_frame", "finish_recording")
            ):
                return None
            return node

    if ast.dump(find_run(source_tree, "LiveEngine")) != ast.dump(
        StripRecording().visit(find_run(copied_tree, "RecordingMixin"))
    ):
        raise RuntimeError(
            "팀 교정 루프가 변경되었습니다. 저장 사본과 원본의 차이를 검토하십시오."
        )
    import live_app.engine as original
    import live_app.server as server_module

    if (
        Path(original.__file__).resolve()
        != (args.correction_root / "live_app/engine.py").resolve()
    ):
        raise RuntimeError("교정 PYTHONPATH와 선택한 원본 경로가 다릅니다.")

    globals().update(
        cv2=original.cv2,
        np=original.np,
        TemporalGate=original.TemporalGate,
        ROOT=original.ROOT,
    )

    class RecordingEngine(RecordingMixin, original.LiveEngine):
        pass

    # Dependency injection for the unchanged make_server factory.
    server_module.LiveEngine = lambda: RecordingEngine(
        args.sessions, args.correction_root
    )
    server = server_module.make_server(args.port, "0", "squats")
    original_handler = server.RequestHandlerClass
    reader = SessionReader(args.sessions)

    class Handler(original_handler):
        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/config":
                    self.reply(
                        200,
                        dict(
                            default_source="0",
                            default_exercise="squats",
                            exercises=original.EXERCISES,
                            full_pose_recording=True,
                            sessions_root=str(args.sessions.resolve()),
                        ),
                    )
                elif parsed.path == "/api/sessions":
                    self.reply(200, dict(sessions=reader.list_sessions()))
                elif parsed.path.startswith("/api/sessions/"):
                    query = parse_qs(parsed.query)
                    self.reply(
                        200,
                        reader.read(
                            parsed.path.rsplit("/", 1)[-1],
                            offset=int(query.get("offset", ["0"])[0]),
                            limit=int(query.get("limit", ["1"])[0]),
                            verify=query.get("verify", ["0"])[0] == "1",
                        ),
                    )
                else:
                    super().do_GET()
            except FileNotFoundError:
                self.reply(404, dict(error="세션이 없습니다."))
            except (ValueError, OSError) as exc:
                self.reply(400, dict(error=str(exc)))

    server.RequestHandlerClass = Handler

    def stop_server(signum, frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    print(f"웹캠 교정·관절 저장 서버: http://127.0.0.1:{args.port}", flush=True)
    with server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


class RecordingMixin:
    def __init__(self, sessions, source):
        super().__init__()
        self.sessions_root = Path(sessions)
        self.source_root = Path(source)
        self.writer = None
        self.want_record = True
        self.session_control = threading.RLock()
        self.recording_error = ""

    def start(self, source, exercise, record=True):
        with self.session_control:
            self.stop()
            self.want_record = record
            self.recording_error = ""
            # Original JSONL stores only successful predictions; our writer records all inputs.
            super().start(source, exercise, False)

    def stop(self):
        with self.session_control:
            super().stop()

    def snapshot(self):
        state = super().snapshot()
        state["full_pose_recording"] = True
        state["record"] = self.want_record if state["running"] else False
        state["recording_error"] = self.recording_error
        writer = self.writer
        if writer:
            state.update(
                session_id=writer.session_id,
                session_path=str(writer.directory),
                saved_frames=writer.count,
                session_finalized=(writer.directory / "summary.json").exists(),
            )
        return state

    def begin_recording(self, exercise):
        self.writer = None
        if not self.want_record:
            return
        checkpoint = self.source_root / (
            "outputs/best.pt"
            if exercise == "squats"
            else f"multi/{exercise}/outputs/best.pt"
        )
        paths = [
            self.source_root / name
            for name in (
                "live_app/engine.py",
                "inference.py",
                "suite_inference.py",
                "features.py",
                "suite_features.py",
                "models.py",
                "suite_config.py",
            )
        ]
        paths.append(checkpoint)
        self.writer = SessionWriter(
            self.sessions_root,
            dict(
                exercise=exercise,
                source=self.snapshot()["source"],
                original_files_sha256={
                    str(p.relative_to(self.source_root)): digest(p)
                    for p in paths
                },
                recording_server_sha256=digest(Path(__file__)),
            ),
        )

    def capture_frame(self, raw, result, stamp, sequence, epoch, width, height):
        if self.writer:
            world = (
                [
                    [p.x, p.y, p.z, p.visibility]
                    for p in result.pose_world_landmarks.landmark
                ]
                if result.pose_world_landmarks
                else None
            )
            self.writer.append(
                raw,
                world,
                stamp,
                sequence,
                epoch,
                width,
                height,
                self.snapshot(),
            )

    def finish_recording(self):
        if self.writer:
            try:
                self.writer.finish(
                    self.snapshot(),
                    (
                        "stop_requested"
                        if self.stop_event.is_set()
                        else "engine_exit"
                    ),
                )
            except Exception as exc:
                self.recording_error = str(exc)
                self.publish(
                    status="error",
                    error=str(exc),
                    message="결과 저장 마무리에 실패했습니다.",
                )

    def run(self, exercise, record):
        import mediapipe as mp
        from suite_inference import FiveExerciseClassifier

        model = None
        pose = None
        log = None
        try:
            self.begin_recording(exercise)
            model = FiveExerciseClassifier().model(exercise)
            required = (
                [11, 12, 23, 24, 25, 26, 27, 28]
                if exercise == "squats"
                else model.saved["required_joints"]
            )
            length = (
                model.length
                if exercise == "squats"
                else model.saved["sequence_length"]
            )
            fps = (
                model.fps if exercise == "squats" else model.saved["sample_fps"]
            )
            gate = TemporalGate(length)
            pose = mp.solutions.pose.Pose(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            if record:
                directory = ROOT / "live_sessions"
                directory.mkdir(exist_ok=True)
                filename = directory / (
                    time.strftime("%Y%m%d_%H%M%S")
                    + "_"
                    + exercise
                    + "_"
                    + str(time.time_ns())[-6:]
                    + ".jsonl"
                )
                log = filename.open("w")
                self.publish(log_path=str(filename))
            self.publish(
                status="connecting",
                message="ESP32-CAM에 연결 중…",
                required_frames=length,
            )
            self.reader.start()
            seen = -1
            epoch = -1
            last_tick = 0.0
            processing_times = deque(maxlen=30)
            while not self.stop_event.is_set():
                seq, current_epoch, item, connected, error, capture_fps = (
                    self.reader.snapshot()
                )
                if self.reset_event.is_set():
                    gate.clear()
                    self.reset_event.clear()
                    pose.reset()
                    last_tick = 0.0
                if current_epoch != epoch:
                    epoch = current_epoch
                    gate.clear()
                    pose.reset()
                    last_tick = 0.0
                    self.publish(
                        result=None, buffer_frames=0, pose_visible=False
                    )
                if not connected or item is None:
                    gate.clear()
                    self.publish(
                        status="connecting",
                        message=error or "영상 연결을 기다리는 중…",
                        error=error,
                        result=None,
                        buffer_frames=0,
                        pose_visible=False,
                        capture_fps=0.0,
                        processing_fps=0.0,
                    )
                    if seen != -2:
                        self.clear_view(
                            "Camera offline - check power / Wi-Fi / URL"
                        )
                        seen = -2
                    self.stop_event.wait(0.1)
                    continue
                frame, stamp = item
                if time.monotonic() - stamp > 1.0:
                    gate.clear()
                    self.publish(
                        status="connecting",
                        message="최신 영상 수신을 기다립니다.",
                        result=None,
                        buffer_frames=0,
                        pose_visible=False,
                    )
                    self.clear_view("Waiting for a fresh camera frame")
                    self.stop_event.wait(0.1)
                    continue
                if seq == seen or stamp - last_tick + 1e-08 < 1 / fps:
                    self.stop_event.wait(0.01)
                    continue
                seen = seq
                last_tick = stamp
                h, w = frame.shape[:2]
                aspect = w / h
                resized = (
                    cv2.resize(
                        frame,
                        (
                            round(w * 480 / max(h, w)),
                            round(h * 480 / max(h, w)),
                        ),
                    )
                    if max(h, w) > 480
                    else frame
                )
                result = pose.process(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
                raw = (
                    np.array(
                        [
                            [l.x, l.y, l.z, l.visibility]
                            for l in result.pose_landmarks.landmark
                        ],
                        np.float32,
                    )
                    if result.pose_landmarks
                    else None
                )
                stage = gate.append(raw, stamp, aspect, required)
                prediction = None
                message = "전신과 필요한 관절이 잘 보이도록 위치를 조정하세요."
                if stage == "ready":
                    prediction = model.predict_landmarks(
                        np.array(gate.points), aspect
                    )
                    if prediction["status"] == "ok":
                        stable = gate.stabilize(prediction)
                        prediction["raw_detected_errors"] = prediction.get(
                            "detected_errors", []
                        )
                        prediction["detected_errors"] = stable
                        mapping = dict(
                            zip(
                                prediction["raw_detected_errors"],
                                prediction.get("feedbacks", []),
                            )
                        )
                        prediction["feedbacks"] = [
                            mapping[t] for t in stable if t in mapping
                        ]
                        prediction["display_smoothing"] = (
                            "At least 2 of the last 3 predictions, also present in current prediction"
                        )
                        stage = "predicting"
                        message = (
                            "선택 오류 감지"
                            if stable
                            else "선택 오류 미검출 · 전체 정상 판정은 아닙니다."
                        )
                        self.publish(
                            predictions=self.snapshot()["predictions"] + 1
                        )
                        if log:
                            log.write(
                                json.dumps(
                                    dict(
                                        time=time.time(),
                                        exercise=exercise,
                                        prediction=prediction,
                                    ),
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            log.flush()
                    else:
                        stage = "pose_not_visible"
                        prediction = None
                        gate.clear()
                elif stage == "warming_up":
                    message = f"자세 수집 중 {len(gate.points)}/{length} · 최초 약 {length / fps:.1f}초 필요"
                processing_times.append(time.monotonic())
                processing_fps = (
                    (len(processing_times) - 1)
                    / (processing_times[-1] - processing_times[0])
                    if len(processing_times) > 1
                    else 0.0
                )
                annotated = frame.copy()
                if result.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        annotated,
                        result.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                    )
                if max(annotated.shape[:2]) > 960:
                    annotated = cv2.resize(
                        annotated,
                        (
                            round(w * 960 / max(h, w)),
                            round(h * 960 / max(h, w)),
                        ),
                    )
                cv2.putText(
                    annotated,
                    f"{exercise.upper()} | {stage} | {len(gate.points)}/{length}",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (50, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                jpeg = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )[1].tobytes()
                with self.lock:
                    if self.stop_event.is_set():
                        break
                    if self.reset_event.is_set():
                        continue
                    if prediction is not None:
                        self.state.update(
                            last_result=copy.deepcopy(prediction),
                            last_prediction_at=time.time(),
                        )
                    self.jpeg = jpeg
                    self.frame_seq += 1
                    self.state.update(
                        status=stage,
                        message=message,
                        error="",
                        result=prediction,
                        buffer_frames=len(gate.points),
                        pose_visible=raw is not None
                        and stage != "pose_not_visible",
                        transport=self.reader.transport,
                        capture_fps=round(capture_fps, 1),
                        processing_fps=round(processing_fps, 1),
                        last_frame_at=time.time(),
                    )
                    self.capture_frame(
                        raw, result, stamp, seq, current_epoch, w, h
                    )
        except Exception as e:
            self.publish(
                running=False,
                status="error",
                error=str(e),
                message="실행 중 오류가 발생했습니다: " + str(e),
                result=None,
                buffer_frames=0,
                pose_visible=False,
            )
            self.clear_view("Error - see the status message")
        finally:
            if self.reader:
                self.reader.stop()
            if pose:
                pose.close()
            if model:
                model.close()
            if log:
                log.close()
            self.finish_recording()


if __name__ == "__main__":
    from GUI.FitHelper.runtime_config import ensure_correction_imports

    args = parser.parse_args()
    ensure_correction_imports(
        args.correction_root, "GUI.FitHelper.direct_correction_server"
    )
    serve(args)
