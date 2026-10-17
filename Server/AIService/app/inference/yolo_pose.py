from ultralytics import YOLO

class YOLOPoseEstimator:
    # 영상이 1초에 20프레임 들어오면 모델 로딩이 반복되는 것을 막기위해 클래스 초기화시 한번만 모델을 로드
    # conf: YOLO가 인정할 Confidence Threshold
    def __init__(self, model_path: str, device: str = 'cuda:0', conf: float = 0.5):
        self.device = device
        self.conf = conf

        print(f'[YOLO Pose] 모델 로딩 시작: {model_path}')
        self.model = YOLO(model_path)
        print(f'[YOLO Pose] 모델 로드 완료: {model_path}')

    def predict(self, frame):
        results = self.model.predict(source=frame, device=self.device, conf=self.conf, verbose=False)
        # 결과가 여러개일 수 있으므로 첫번째 결과만 사용
        result = results[0]
        # 결과를 시각화하여 프레임에 표시
        annotated_frame = result.plot()
        # result를 가져오는 이유는 AI 분석에 데이터를 사용하기 위해서
        return annotated_frame, result