"""정면 자세 평가 전용 움직임 관측. 횟수/Rep 이벤트를 생성하지 않습니다."""


class FrontSquatMotion:
    def __init__(self, config):
        self.cfg = config
        self.reset()

    def reset(self):
        self.last = self.baseline = self.candidate = self.since = None
        self.samples = 0
        self.active = False
        self.started = None
        self.metrics = {}

    def observe(self, pose, timestamp):
        if self.last is not None:
            if timestamp <= self.last or timestamp-self.last > self.cfg.squat_pose_grace_sec:
                self.reset()
        pose.require(23,24,25,26,27,28)
        lengths = [pose.distance(h,k)+pose.distance(k,a) for h,k,a in ((23,25,27),(24,26,28))]
        scale = sum(lengths)/2
        if scale < .15:
            raise ValueError('front_motion_degenerate_legs')
        extension = sum(pose.xy(a)[1]-pose.xy(h)[1] for h,a in ((23,27),(24,28)))/(2*scale)
        knee = sum(pose.angle(h,k,a) for h,k,a in ((23,25,27),(24,26,28)))/2
        self.last = timestamp
        self.metrics = dict(extension_ratio=extension, knee_angle=knee, baseline=self.baseline)
        if self.baseline is None:
            # 먼저 안정된 입위를 관측합니다. 웅크린 시작 자세를 기준으로 삼지 않습니다.
            if knee < 155 or extension < .85:
                self.candidate = self.since = None
                self.samples = 0
                return None
            if self.candidate is None or abs(extension-self.candidate) > .03:
                self.candidate,self.since,self.samples = extension,timestamp,1
            else:
                self.samples += 1
            if self.samples >= 3 and timestamp-self.since >= .3:
                self.baseline = self.candidate
                return 'standing'
            return None
        drop = self.baseline-extension
        self.metrics.update(baseline=self.baseline, hip_drop_ratio=drop)
        if self.active:
            if timestamp-self.started > self.cfg.max_cycle_sec:
                self.reset()
                return None
            if drop <= .035:
                self.active = False
                self.started = None
        elif drop >= .08:
            self.active,self.started = True,timestamp
        return 'descending' if self.active else 'standing'
