import time

import cv2
import mediapipe as mp

class MediapipePoseEstimator:
    def __init__(
        self,
        model_path: str,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        print(f'[MediaPipe Pose] 모델 로딩 시작: {model_path}')
        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path,
            delegate=mp.tasks.BaseOptions.Delegate.GPU
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
        self.start_time = time.monotonic()

        print(f'[MediaPipe Pose] 모델 로딩 완료: {model_path}')

    def predict(self, frame):
        # OpenCV는 BGR 형식이므로 MediaPipe Pose가 요구하는 RGB 형식으로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # MediaPipe Pose는 비디오 입력을 처리할 때 각 프레임의 타임스탬프를 필요로 함
        timestamp_ms = int((time.monotonic() - self.start_time) * 1000)
        # OpenCV로 한 프레임씩 가져오고 있기 때문에 동기식 추론
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        annotated_frame = frame.copy()

        if result.pose_landmarks:
            self._draw_landmarks(annotated_frame, result.pose_landmarks[0])

        return annotated_frame, result

    def _draw_landmarks(self, frame, landmarks):
        height, width, _ = frame.shape

        for landmark in landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    def close(self):
        self.landmarker.close()