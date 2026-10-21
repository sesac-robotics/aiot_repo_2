import math

# MediaPipe Pose Landmarker 33개 관절 이름
LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

def extract_mediapipe_keypoints(result):
    '''
    MediaPipe Pose Landmarker Result에서 첫 번재 사람의 관절 좌표 추출

    반환값: list[dict]
    '''

    if not result.pose_landmarks:
        return []

    # 현재 모델은 num_poses=1
    landmarks = result.pose_landmarks[0]

    keypoints = []

    for idx, landmark in enumerate(landmarks):
        # 비정상적인 좌표 제외
        if not all(
            math.isfinite(value)
            for value in (landmark.x, landmark.y, landmark.z)
        ):
            continue

        keypoints.append({
            'id': idx,
            'name': LANDMARK_NAMES[idx],
            'x': float(landmark.x),
            'y': float(landmark.y),
            'z': float(landmark.z),
            'visibility': (
                float(landmark.visibility)
                if landmark.visibility is not None
                else None
            ),
            'presence': (
                float(landmark.presence)
                if landmark.presence is not None
                else None
            )
        })

    return keypoints