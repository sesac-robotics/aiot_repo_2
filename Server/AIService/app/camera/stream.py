import cv2

'''
실제 서버에서는 카메라를 한 번 읽고 끝내지 않고 카메라 연결 상태를 계속 유지해야 함
CameraStream 클래스를 ㅌ오해 카메라 연결 자체를 관리하도록 구현
'''
class CameraStream:
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.capture = None

    def connect(self):
        self.capture = cv2.VideoCapture(self.stream_url)

        if not self.capture.isOpened():
            raise ConnectionError(f'카메라에 연결할 수 없습니다.: {self.stream_url}')
        print(f'[Camera] 연결 성공: {self.stream_url}')

    def read(self):
        if self.capture is None:
            raise RuntimeError('카메라가 연결되지 않았습니다.')

        success, frame = self.capture.read()

        if not success:
            return None

        return frame

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

        print('[Camera] 연결 종료')