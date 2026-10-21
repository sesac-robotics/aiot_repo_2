import cv2
import time
import numpy as np
import mediapipe as mp

# 부위별 Keypoint 색상 (OpenCV BGR)

KEYPOINT_COLORS = {
    'face': (0, 165, 255),
    'upper': (255, 220, 0),
    'lower': (70, 210, 70)
}
# 부위별 Skeleton 연결선 색상
LINE_COLORS = {
    'face': (0, 110, 220),
    'upper': (220, 150, 0),
    'lower': (40, 150, 40)
}

FACE_LANDMARK_IDS = set(range(11))

class MediapipePoseEstimator:
    def __init__(
        self,
        model_path: str,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_visibility=0.35,
        delegate='gpu'
    ):
        # 서비스가 주입하는 계측기이며 GPU 객체 생성과는 무관합니다.
        self.metrics = None
        self.min_visibility = min_visibility
        self.last_timestamp_ms = -1

        delegate = delegate.lower()
        if delegate == 'gpu':
            mp_delegate = mp.tasks.BaseOptions.Delegate.GPU
        elif delegate == 'cpu':
                mp_delegate = mp.tasks.BaseOptions.Delegate.CPU
        else:
            raise ValueError(f'지원하지 않는 MediaPipe delagtae: {delegate}')
        
        print(f'[MediaPipe Pose] 모델 로딩 시작: {model_path}')
        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path,
            delegate=mp_delegate,
        )
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            # 사람 한 명만 추적하도록 설정
            num_poses=1,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        # 시스템 시계가 바뀌더라도 값이 뒤로 가지 않는 단조 증가 시계이기 때문에 monotonic()을 사용
        # self.start_time = time.monotonic()

        # MediaPipe에서 제공하는 공식 관절 연결 정보
        self.connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

        print(
            f'[MediaPipe Pose] 모델 로딩 완료: {model_path}'
            f'(delegate={delegate})'    
        )

    def _get_body_part(self, index):
        # MediaPipe Pose의 33개 관절 인덱스 분류
        if 0 <= index <= 10:
            return 'face'
        elif 11 <= index <= 22:
            return 'upper'
        else:
            return 'lower'

    def _get_connection_part(self, start, end):
        start_part = self._get_body_part(start)
        end_part =self._get_body_part(end)

        # 동일 부위의 연결선은 해당 부위 색상 사용
        if start_part == end_part:
            return start_part

        # 상체와 골반 사이 몸통 연결선
        return 'upper'

    def _is_valid_landmark(self, landmark):
        visibility = getattr(landmark, 'visibility', None)

        presence = getattr(landmark, 'presence', None)

        if visibility is not None and visibility < self.min_visibility:
            return False

        # 화면 밖 좌표는 시각화하지 않음
        return(
            np.isfinite(landmark.x)
            and np.isfinite(landmark.y)
            and 0 <= landmark.x <= 1
            and 0 <= landmark.y <= 1
        )

    def _draw_skeleton(self, frame, landmarks):
        height, width = frame.shape[:2]

        # 관절의 화면 좌표 및 유효성 계산
        points = {}

        for idx, landmark in enumerate(landmarks):
            if not self._is_valid_landmark(landmark):
                continue
            # 얼굴 Keypoin 제외
            if idx in FACE_LANDMARK_IDS:
                continue

            x = min(int(landmark.x * width), width - 1)
            y = min(int(landmark.y * height), height - 1)

            points[idx] = (x, y)

        # 1. Skeleton 연결선 먼저 그리기
        for connection in self.connections:
            start = connection.start
            end = connection.end

            if start not in points or end not in points:
                continue
            # 얼굴과 연결된 선은 표시하지 않음
            if start in FACE_LANDMARK_IDS or end in FACE_LANDMARK_IDS:
                continue

            part = self._get_connection_part(start, end)
            color = LINE_COLORS[part]

            cv2.line(
                frame, 
                points[start], 
                points[end], 
                color, 
                thickness=2, 
                lineType=cv2.LINE_AA
            )

        # 2. 연결선 위에 Keypoint 그리기
        for idx, point in points.items():
            part = self._get_body_part(idx)
            color = KEYPOINT_COLORS[part]

            # 흰색 외곽선
            cv2.circle(
                frame,
                point, 
                radius=5,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA
            )

            # 부위별 색상의 관절 점
            cv2.circle(
                frame,
                point,
                radius=3,
                color=color,
                thickness=-1,
                lineType=cv2.LINE_AA
            )

        return frame

    def predict(self, frame):
        # OpenCV는 BGR 형식이므로 MediaPipe Pose가 요구하는 RGB 형식으로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # MediaPipe Pose는 비디오 입력을 처리할 때 각 프레임의 타임스탬프를 필요로 함
        timestamp_ms = time.monotonic_ns() // 1_000_000
        timestamp_ms = max(timestamp_ms, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms
    
        # OpenCV로 한 프레임씩 가져오고 있기 때문에 동기식 추론
        # 동기식 MediaPipe 호출만 측정: 색 변환/스켈레톤 그리기는 제외합니다.
        detect_start = time.perf_counter()
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if self.metrics is not None:
            self.metrics.record_duration(
                "mediapipe_inference", (time.perf_counter() - detect_start) * 1000
            )
        # 원본 프레임을 복사해서 시각화
        annotated_frame = frame.copy()

        if result.pose_landmarks:
            for landmarks in result.pose_landmarks:
                annotated_frame = self._draw_skeleton(annotated_frame, landmarks)

        # 기존 AI Server 인터페이스 유지
        return annotated_frame, result

    # def _draw_landmarks(self, frame, landmarks):
    #     height, width, _ = frame.shape

    #     for landmark in landmarks:
    #         x = int(landmark.x * width)
    #         y = int(landmark.y * height)

    #         cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    def close(self):
        self.landmarker.close()
        print('[MediaPipe] 모델 종료')