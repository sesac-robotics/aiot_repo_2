import cv2
import time
import threading

'''
실제 서버에서는 카메라를 한 번 읽고 끝내지 않고 카메라 연결 상태를 계속 유지해야 함
CameraStream 클래스를 ㅌ오해 카메라 연결 자체를 관리하도록 구현
'''
class CameraStream:
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.capture = None

        # 카메라에서 받은 최신 프레임을 저장하는 변수
        self.latest_frame = None
        # 새로운 프레임이 들어왔는지 구분하기 위한 변수
        self.frame_id = 0
        # YOLO가 마지막으로 가져간 프레임 번호
        self.last_read_id = 0

        # Condition은 여러 스레드가 공유하는 데이터의 상태를 확인하고
        # 특정 조건이 충족될 때까지 기다리거나 다른 스레드에 알리는 동기화 도구
        self.condition = threading.Condition()
        # Event는 스레드 사이에서 특정 상태를 공유하는 사용하는 객체
        # 카메라 수신 스레드에 종료 신호를 전달하는 역할
        self.stop_event = threading.Event()
        self.thread = None

    def connect(self):
        if self.capture is not None:
            return
        
        capture = cv2.VideoCapture(self.stream_url)

        if not capture.isOpened():
            capture.release()
            raise ConnectionError(f'카메라에 연결할 수 없습니다.: {self.stream_url}')

        # 새로운 연결을 시작하므로 기존 프레임과 프레임 번호를 초기화
        self.capture = capture
        self.stop_event.clear()
        self.latest_frame = None
        self.frame_id = 0
        self.last_read_id = 0

        # 별도 스레드에서 영상 수신
        self.thread = threading.Thread(
            # 새로운 스레드에서 실행할 함수를 지정
            target=self._capture_loop,
            # 해당 함수에 전달할 인자
            args=(capture,),
            # 데몬 스레드로 생성한다는 의미, 데몬이 아닌 스레드가 모두 종료되면 프로그램 종료를 막지 않음
            daemon=True,
        )
        # 실제 새로운 스레드를 시작
        self.thread.start()

        print(f'[Camera] 연결 성공: {self.stream_url}')

    def _capture_loop(self, capture):
        # 종료 요청이 들어오기 전까지 계속 프레임을 수신
        while not self.stop_event.is_set():
            success, frame = capture.read()
            
            # 프레임 수신 실패 처리
            # 프레임 수신에 실패하면 0.1초 동안 대기한 뒤 다시 시도
            if not success:
                self.stop_event.wait(0.1)
                continue

            # 가장 최근 프레임만 보관
            with self.condition:
                self.latest_frame = frame
                self.frame_id += 1
                self.condition.notify_all()

    def read(self):
        if self.capture is None:
            raise RuntimeError('카메라가 연결되지 않았습니다.')

        with self.condition:
            # 새로운 프레임이 도착할 때까지 최대 1초 대기
            # wait_for()는 지정한 조건이 만족될 때까지 대기하는 함수
            self.condition.wait_for(
                lambda: (
                    # 새로운 프레임이 도착했는지 확인하는 조건
                    self.frame_id > self.last_read_id
                    # 카메라 수신 종료 요청이 발생했는지 확인하는 조건
                    or self.stop_event.is_set()
                ),
                timeout=1.0
            )

            # 프레임이 없으면 None 반환
            if (
                self.stop_event.is_set()
                or self.frame_id == self.last_read_id
            ):
                return None

            # 일근 프레임 번호 갱신
            self.last_read_id = self.frame_id

            # 수신 스레드와 프레임 데이터 분리
            # .copy()를 사용하는 이유는 frame과 latest_frame이 같은 배열을 참조하는 것을 막기 위해
            return self.latest_frame.copy()

    def release(self):
        # 카메라 수신 스레드에 종료 신호를 보냄
        self.stop_event.set()

        # read()에서 새로운 프레임을 기다리고 있는 스레드한테 종료 신호를 확인할 수 있도록 함
        with self.condition:
            self.condition.notify_all()

        # join()은 해당 스레드가 종료될 때까지 기다리는 함수
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            # 2초 뒤에도 스레드가 살아 있다면 경고 메시지를 출력
            if self.thread.is_alive():
                print('[Camera] 수신 스레드 종료 지연')

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        print('[Camera] 연결 종료')