import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'output/live-webcam-20261023'))
from test_shoulder_candidate import pose
from anatomical_turn_features_v1 import extract
class Tests(unittest.TestCase):
    def seq(self):return list(enumerate([pose(True).copy() for _ in range(5)]+[pose().copy() for _ in range(5)]+[pose(True).copy() for _ in range(5)]))
    def event(self):return dict(protocol='shoulder_press',laterality='bilateral',posture='standing',peak_frames=[0,4],turn_frames=[5,9],next_ascent_frames=[10,14],closure='observed_next_ascent')
    def runx(self,s=None,e=None,frames=None):
        s=self.seq() if s is None else s
        return extract(s,self.event() if e is None else e,10,native_frame_numbers=[t for t,_ in s] if frames is None else frames)
    def test_top_start_without_ready(self):
        r=self.runx();self.assertEqual(r['status'],'features_ready');self.assertFalse(r['ready_reference_used']);self.assertIsNone(r['grade']);self.assertFalse(r['tracking_QA_proven'])
    def test_clipped_no_following_is_not_failure(self):self.assertEqual(self.runx(self.seq()[:10])['status'],'held')
    def test_absent_native_clock_held(self):self.assertEqual(extract(self.seq(),self.event(),10)['reason'],'actual_native_frame_alignment_required')
    def test_misordered_windows(self):
        e=self.event();e['next_ascent_frames']=[8,14];self.assertEqual(self.runx(e=e)['reason'],'peak_turn_next_ascent_order_invalid')
    def test_missing_inside_phase_not_imputed(self):
        s=self.seq();s[7][1][11]=np.nan;self.assertEqual(self.runx(s)['status'],'held')
    def test_sparse_time_not_compacted(self):self.assertEqual(self.runx(self.seq()[::2])['status'],'held')
    def test_nonpress_not_admitted(self):
        e=self.event();e['protocol']='arm_extension';self.assertEqual(self.runx(e=e)['status'],'held')
    def test_side_not_bilateral_target(self):
        e=self.event();e['laterality']='near_arm';self.assertEqual(self.runx(e=e)['status'],'held')
    def test_missing_hip_or_border_not_grade(self):
        s=self.seq();s[7][1][23]=[0,0];self.assertEqual(self.runx(s)['status'],'held')
    def test_last_rack_without_next_ascent(self):
        e=self.event();e.update(closure='observed_rack',rack_visible_stable=True);r=self.runx(self.seq()[:10],e);self.assertEqual(r['status'],'features_ready')
    def test_no_descent_not_event(self):
        self.assertEqual(self.runx([(t,pose(True)) for t in range(15)])['reason'],'observed_descent_not_established')
    def test_no_ascent_not_partial_closure(self):
        s=self.seq()
        for t in range(10,15):s[t]=(t,pose())
        self.assertEqual(self.runx(s)['reason'],'observed_next_ascent_not_established')
    def test_anatomical_vs_relative_schema_distinct(self):
        from observation_features_v4 import RETURN_SCHEMA
        self.assertNotEqual(self.runx()['version'],RETURN_SCHEMA)
if __name__=='__main__':unittest.main()
