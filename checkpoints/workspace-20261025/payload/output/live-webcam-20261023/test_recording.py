"""실제 인물 촬영 없이 저장·취소·시간 제한 계약을 검사한다."""
import json
import tempfile
import time
import unittest
from pathlib import Path
import cv2
import numpy as np
from recording import Recorder, RUBRIC

class RecordingTest(unittest.TestCase):
    def test_cancel_writes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            r=Recorder(root);r.begin({'source_commit':'test'})
            r.stop()
            self.assertEqual(list(Path(root).iterdir()),[])
            self.assertFalse(r.active)

    def test_clean_video_and_pose_preserve_missing_slots(self):
        with tempfile.TemporaryDirectory() as root:
            r=Recorder(root);r.begin({'source_commit':'synthetic-test-only'})
            r.start_at=time.monotonic()-1
            frame=np.zeros((48,64,3),np.uint8)
            xy=np.full((33,2),np.nan)
            for tick in [100,101,103]: r.push(frame,xy,tick,time.monotonic())
            r.stop()
            folder=next(Path(root).iterdir())
            pose=np.load(folder/'pose.npz')
            self.assertEqual(pose['frame_numbers'].tolist(),[0,1,3])
            self.assertTrue(np.isnan(pose['image_xy']).all())
            meta=json.loads((folder/'metadata.json').read_text())
            self.assertTrue(meta['source_scores_hidden'])
            self.assertEqual(meta['frames'],3)
            cap=cv2.VideoCapture(str(folder/'blind.avi'))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),3);cap.release()
            labels=json.loads((folder/'labels.draft.json').read_text())
            self.assertEqual(labels['repetitions'],[])
            self.assertFalse(labels['frozen'])

    def test_timeout_and_no_double_begin(self):
        with tempfile.TemporaryDirectory() as root:
            r=Recorder(root);r.begin({})
            with self.assertRaises(ValueError):r.begin({})
            r.start_at=time.monotonic()-61
            r.push(np.zeros((48,64,3),np.uint8),np.zeros((33,2)),0,time.monotonic())
            self.assertFalse(r.active)
            self.assertEqual(list(Path(root).iterdir()),[])

    def test_checklist_has_no_invented_weights(self):
        self.assertTrue(all('weight' not in i for i in RUBRIC['items']))
        self.assertEqual(len({i['id'] for i in RUBRIC['items']}),6)

if __name__=='__main__':unittest.main()
