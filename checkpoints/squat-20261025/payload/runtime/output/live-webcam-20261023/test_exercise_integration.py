"""운동 후보 선택에서 배포 점수·미평가 표시까지 검증한다."""
import unittest
import numpy as np
from server import make_tracker,score_samples
from test_shoulder_candidate import pose
from shoulder_candidate import ShoulderTracker
from movement import MovementTracker

class ExerciseIntegrationTests(unittest.TestCase):
    def test_factory_selects_shoulder_only_and_rejects_unsupported_view(self):
        self.assertIsInstance(make_tracker('shoulder','front'),ShoulderTracker)
        self.assertIsInstance(make_tracker('squats','front'),MovementTracker)
        with self.assertRaises(ValueError):make_tracker('shoulder','side')

    def test_completed_shoulder_pipeline_keeps_single_grade_ai_unassessed(self):
        tracker=make_tracker('shoulder');segment=None
        for tick,xy in enumerate([pose()]*6+[pose(True)]*4+[pose()]*3):
            event=tracker.push(tick,xy)
            if event:segment=event
        self.assertEqual(tracker.count,1)
        result=score_samples(segment,640,480,'shoulder','front')
        shown=result['score_100'];items={r['id']:r for r in shown['items']}
        self.assertIsNotNone(items['overhead_reach']['ai_points'])
        self.assertIsNone(items['lower_return']['ai_points'])
        self.assertIsNone(items['frontal_trunk_hold']['ai_points'])
        self.assertIsNone(shown['ai']['total'])
        self.assertEqual(shown['ai']['assessed_weight'],55)
        self.assertEqual(items['lower_return']['title'],'하단 복귀')
        self.assertIn('정상 한 등급',items['lower_return']['reasons'][0])

    def test_current_buffer_does_not_include_previous_repetition(self):
        tracker=make_tracker('shoulder');events=[]
        seq=([pose()]*6+[pose(True)]*4+[pose()]*3)*2
        for tick,xy in enumerate(seq):
            event=tracker.push(tick,xy)
            if event:
                events.append(event)
                self.assertFalse(tracker.active);self.assertEqual(tracker.buffer,[])
        self.assertEqual(len(events),2)
        self.assertGreater(events[1][0][0],events[0][-1][0])

if __name__=='__main__':unittest.main()
