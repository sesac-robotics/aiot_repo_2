"""점수를 포함하지 않는 로컬 정면 스쿼트 촬영과 독립 라벨 양식."""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

from squat_criteria import RUBRIC

class Recorder:
    """명시적 시작 후 5초 대기하고 최대 60초만 로컬 저장한다."""
    def __init__(self, root):
        self.root = Path(root)
        self.phase = 'idle'
        self.last = None
        self.writer = None
        self.samples = []
        self.frames = []

    @property
    def active(self):
        return self.phase in ('countdown', 'recording')

    def status(self):
        now = time.monotonic()
        return dict(phase=self.phase, countdown=max(0, round(self.start_at-now, 1)) if self.active else 0,
                    seconds=round(max(0,now-self.start_at),1) if self.active else 0,
                    frames=len(self.samples), last=self.last)

    def begin(self, provenance):
        if self.active:
            raise ValueError('이미 촬영 대기·촬영 중입니다')
        self.provenance = dict(provenance)
        self.start_at = time.monotonic()+5
        self.phase = 'countdown'
        self.samples = []
        self.frames = []
        self.writer = None
        self.last = None

    def push(self, frame, xy, tick, captured_at):
        if not self.active or time.monotonic() < self.start_at:
            return
        if time.monotonic()-self.start_at >= 60:
            self.stop('60초 자동 종료')
            return
        if self.writer is None:
            self.folder = self.root / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            self.folder.mkdir(parents=True)
            h,w = frame.shape[:2]
            self.writer = cv2.VideoWriter(str(self.folder/'blind.avi'),cv2.VideoWriter_fourcc(*'MJPG'),10.,(w,h))
            if not self.writer.isOpened():
                self.phase = 'error'
                raise ValueError('영상 저장 장치를 열지 못했습니다')
            self.shape = (w,h)
        self.phase = 'recording'
        self.writer.write(frame)
        self.samples.append(xy.copy())
        self.frames.append((tick, captured_at-self.start_at))

    def stop(self, reason='사용자 종료'):
        if not self.active:
            return self.status()
        if self.writer is None:
            self.phase = 'idle'
            self.last = {'status': '촬영 전 취소 · 저장 없음'}
            return self.status()
        self.writer.release()
        self.writer = None
        ticks = np.array([f[0] for f in self.frames],dtype=np.int64)
        stamps = np.array([f[1] for f in self.frames])
        np.savez_compressed(self.folder/'pose.npz',image_xy=np.array(self.samples),frame_numbers=ticks-ticks[0],timestamps=stamps,fps=10.,width=self.shape[0],height=self.shape[1])
        video_hash = hashlib.sha256((self.folder/'blind.avi').read_bytes()).hexdigest()
        metadata = dict(self.provenance, rubric_version=RUBRIC['version'],participant='P001',split='unassigned',
                        recording_group=self.folder.name,video_sha256=video_hash,frames=len(ticks),
                        duration_seconds=round(float(stamps[-1]-stamps[0]),3),stop_reason=reason,
                        video='blind.avi',pose='pose.npz',video_nominal_fps=10,
                        timing_note='실제 시간은 pose.npz timestamps 기준. 누락 시 영상 재생 시간이 짧아질 수 있음.',
                        source_scores_hidden=True,face_mosaic='관측된 대상 얼굴만 적용; 미검출 프레임은 얼굴 보호를 보장하지 않음')
        for name,obj in [('metadata.json',metadata),('rubric.json',RUBRIC),('labels.draft.json',{
            'rubric_version':RUBRIC['version'],'video_sha256':video_hash,'label_origin':'direct_visual_draft',
            'expert_validated':False,'frozen':False,'repetitions':[],
            'instructions':'각 회차 start_frame/end_frame(영상 0기반), 각 항목 grade 0/1/2 또는 null, 시각 근거를 기록. 점수 출력 열람 전 고정.'})]:
            (self.folder/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
        self.phase = 'saved'
        self.last = {'status':'저장 완료 · 독립 시각 라벨 대기','folder':str(self.folder),'frames':len(ticks)}
        return self.status()
