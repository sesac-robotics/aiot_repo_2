import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'output/live-webcam-20261023'))
from test_shoulder_candidate import pose
from return_context_v1 import ReturnContextQueue
class Tests(unittest.TestCase):
    def seed(self):return list(enumerate([pose().copy() for _ in range(6)]+[pose(True).copy() for _ in range(4)]+[pose().copy() for _ in range(2)]))
    def test_waits_for_actual_frame(self):
        q=ReturnContextQueue();self.assertIsNone(q.submit('a',self.seed()));r=q.push(12,pose())[0]
        self.assertEqual(r['features']['return_ticks'],[10,11,12]);self.assertEqual(r['added_actual_ticks'],[12]);self.assertIsNone(r['score'])
    def test_recording_end_is_never_failure(self):
        q=ReturnContextQueue();q.submit('a',self.seed());r=q.finish()[0]
        self.assertEqual(r['features']['status'],'held');self.assertEqual(r['context_reason'],'recording_ended_before_return_context_closed');self.assertIsNone(r['grade'])
    def test_gap_not_filled(self):
        q=ReturnContextQueue();q.submit('a',self.seed());r=q.push(14,pose())[0]
        self.assertEqual(r['features']['status'],'held');self.assertEqual(r['added_actual_ticks'],[14])
    def test_frame_after_deadline_not_used(self):
        q=ReturnContextQueue();q.submit('a',self.seed());r=q.push(15,pose())[0]
        self.assertEqual(r['evaluated_at_tick'],11);self.assertEqual(r['added_actual_ticks'],[]);self.assertEqual(r['features']['status'],'held')
    def test_context_limit_not_infinite_future(self):
        with self.assertRaises(ValueError):ReturnContextQueue(20)
    def test_already_closed_has_no_delay(self):
        q=ReturnContextQueue();s=self.seed()+[(12,pose())];r=q.submit('a',s)
        self.assertEqual(r['features']['status'],'features_ready');self.assertFalse(q.pending)
    def test_zero_context_preserves_hold(self):
        q=ReturnContextQueue(0);self.assertEqual(q.submit('a',self.seed())['features']['status'],'held')
    def test_duplicate_slots_rejected(self):
        q=ReturnContextQueue();q.submit('a',self.seed())
        with self.assertRaises(ValueError):q.push(11,pose())
    def test_missing_joint_not_reconstructed(self):
        q=ReturnContextQueue();q.submit('a',self.seed());p=pose();p[23]=np.nan
        out=q.push(12,p)+q.push(13,pose())+q.push(14,pose())+q.finish()
        self.assertTrue(all(r['features']['status']=='held' for r in out))
if __name__=='__main__':unittest.main()
