import sys,unittest
from pathlib import Path
O=Path(__file__).resolve().parent;A=O.parents[1]/'output/live-webcam-20261023';sys.path[:0]=[str(O),str(A)]
import movement_stage_candidate as stage
sys.modules['movement_candidate']=stage
import test_motion_recovery as base
from test_motion_recovery import pose
import numpy as np
class StageTest(unittest.TestCase):
 def feed(self,seq):return base.MotionRecoveryTest().feed(seq)
 def test_actual_core_raw_samples_preserve_missing_knees(self):
  seq=[pose()]*10+[pose(.3)]*6+[pose(.3,0)]*4+[pose(.3)]*8+[pose()]*5;t,segs=self.feed(seq)
  self.assertEqual(t.count,1);raw=dict(segs[0]);self.assertTrue(all(np.isnan(raw[n][[25,26]]).all()for n in range(16,20)))
  r=t.completion_receipts[0]['observation'];self.assertEqual(r['detector_core_coverage'],1);self.assertEqual(r['actual_time_slot_coverage'],1);self.assertLess(r['knee_phase_evidence_coverage'],1)
 def test_ready_reference_survives_short_entry_gap_without_phase_guess(self):
  t=stage.MovementTracker('squats','front')
  for n in range(10):t.push(n,pose())
  for n in range(10,15):
   t.push(n,pose(.3,0));self.assertFalse(t.active);self.assertFalse(t.has_bottom)
  for n in [15,16]:t.push(n,pose(.3));self.assertFalse(t.active)
  for n in [17,18,19]:t.push(n,pose(.3))
  self.assertTrue(t.has_bottom)
  for n in [20,21,22]:t.push(n,pose())
  self.assertEqual(t.count,1)
 def test_final_standing_arm_movement_gap_not_a_rep(self):
  t,_=self.feed([pose()]*10+[pose(0,0)]*4+[pose()]*15);self.assertEqual(t.count,0)
 def test_score_gate_does_not_inherit_core_coverage(self):
  from front_candidate import vector
  seq=[pose()]*10+[pose(.3)]*3+[pose(.3,0)]*6+[pose(.3)]*3+[pose()]*3;t,segs=self.feed(seq)
  self.assertEqual(t.count,1);r=t.completion_receipts[0]['observation'];self.assertEqual(r['detector_core_coverage'],1);self.assertLess(r['knee_phase_evidence_coverage'],.8)
  with self.assertRaises(ValueError):vector(np.array([p for _,p in segs[0]]))
 def test_unobserved_core_cannot_complete_fake_boundaries(self):
  missing=pose(.3);missing[[11,12,23,24]]=np.nan
  t,_=self.feed([pose()]*10+[missing]*10+[pose()]*10);self.assertEqual(t.count,0)
 def test_single_unobserved_core_breaks_return_streak(self):
  t=stage.MovementTracker('squats','front')
  for n in range(10):t.push(n,pose())
  for n in [10,11,12]:t.push(n,pose(.3))
  t.push(13,pose());t.push(14,pose())
  bad=pose();bad[23]=np.nan;t.push(15,bad)
  t.push(16,pose());self.assertEqual(t.count,0);self.assertEqual(t.returned,1)
  t.push(17,pose());t.push(18,pose());self.assertEqual(t.count,1)
 def test_single_missing_core_breaks_bottom_streak(self):
  t=stage.MovementTracker('squats','front')
  for n in range(10):t.push(n,pose())
  t.push(10,pose(.3));t.push(11,pose(.3));bad=pose(.3);bad[23]=np.nan;t.push(12,bad)
  t.push(13,pose(.3));self.assertFalse(t.has_bottom)
if __name__=='__main__':
 suite=unittest.TestSuite()
 for cls in [base.MotionRecoveryTest,base.TimeAxisTest,StageTest]:
  for name in unittest.defaultTestLoader.getTestCaseNames(cls):
   if cls==base.MotionRecoveryTest and name in ['test_short_knee_gap_recovers_without_imputation','test_coverage_uses_actual_missing_slots']:continue
   suite.addTest(cls(name))
 r=unittest.TextTestRunner().run(suite);sys.exit(not r.wasSuccessful())
