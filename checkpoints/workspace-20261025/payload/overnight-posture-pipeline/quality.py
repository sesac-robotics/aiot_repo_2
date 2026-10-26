"""운동별 핵심 부위의 관측 가능성을 검사한다. 촬영 방향으로 배제하거나 자세 점수를 계산하지 않는다."""
import cv2
import numpy as np
from common import BASE

VERSION='20261020-v3-exercise-specific'
GROUPS={
 'pushups':[(11,12),(13,14),(23,24),(27,28)],
 'squats':[(11,12),(23,24),(25,26),(27,28)],
 'plank':[(11,12),(23,24),(27,28)],
 'ytw':[(11,12),(13,14),(15,16),(23,24)],
 'shoulder':[(11,12),(13,14),(15,16),(23,24)]}


def observation_ok(visible_frames,decoded_frames):
    """4시점 중 핵심 부위가 보이는 2시점 이상인지 반환한다. 자세 정확도 판단은 아니다."""
    return decoded_frames==4 and visible_frames>=2


class VisibilityGate:
    """측면 강제 없이 운동별 핵심 부위의 화면 내 관측 여부를 반환한다."""

    def __init__(self):
        """Heavy GPU 검출기를 생성한다. GPU 초기화 실패는 호출자에게 전달한다."""
        import sys
        sys.path.insert(0,str(BASE/'mediapipe-pilot/qevd_coach'))
        import pose_features
        self.detector=pose_features.make_detector('heavy')

    def check(self,row):
        """클립의 네 시점을 검사하고 부위별 실패 원인·비율을 반환한다.

        Args:
            row: 영상 경로, 운동명 및 선택적인 시작·끝 프레임이 포함된 객체.
        Returns:
            관측 가능 여부, 프레임 수, 그룹별 관측 수. 확률·좌표는 무차원이다.
        Raises:
            ValueError: 알려지지 않은 운동명.
        """
        import mediapipe as mp
        cap=cv2.VideoCapture(row['path']);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));good=0;decoded=0
        groups=GROUPS[row['exercise']];group_counts=[0]*len(groups)
        first=int(row.get('start_frame',0));last=min(count-1,int(row.get('end_frame',count-1)))
        try:
            for index in np.linspace(first,max(first,last),4).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,frame=cap.read()
                if not ok:continue
                decoded+=1;h,w=frame.shape[:2];scale=min(1,640/max(h,w));frame=cv2.resize(frame,(int(w*scale),int(h*scale)))
                result=self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))))
                if not result.pose_landmarks:continue
                points=result.pose_landmarks[0]
                visible=[any(points[k].visibility>=.3 and points[k].presence>=.5 and 0<=points[k].x<=1 and 0<=points[k].y<=1 for k in group) for group in groups]
                group_counts=[n+int(v) for n,v in zip(group_counts,visible)]
                good+=int(all(visible))
        finally:cap.release()
        return dict(visible=observation_ok(good,decoded),valid_frames=good,decoded_frames=decoded,required_body_groups=groups,group_observed_frames=group_counts,side_view_required=False,quality_version=VERSION,purpose='visibility_only_not_posture_score',reason='관측 가능' if observation_ok(good,decoded) else '핵심 부위 미관측 또는 읽기 부족')

    def close(self):
        """GPU 검출기를 닫아 메모리를 반환한다."""
        self.detector.close()
