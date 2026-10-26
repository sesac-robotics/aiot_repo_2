import sys,unittest
from pathlib import Path
import numpy as np
A=Path('/home/jayden/SeSACPAI-Mid/output/live-webcam-20261023');sys.path.insert(0,str(A))
from movement_candidate import MovementTracker
from test_movement import squat

def pose(drop=0,knees=2):
 p=squat();p[[11,12,23,24],1]+=100*drop
 if knees==0:p[[25,26]]=np.nan
 elif knees==1:p[25]=np.nan
 return p
class MotionRecoveryTest(unittest.TestCase):
 def feed(self,seq):
  t=MovementTracker('squats','front');results=[]
  for tick,p in enumerate(seq):
   seg=t.push(tick,p)
   if seg:results.append(seg)
  return t,results
 def test_brief_excursion_not_completed(self):
  for d in [.085,.089,.095,.4]:
   for duration in [1,2]:
    with self.subTest(drop=d,slots=duration):
     t,_=self.feed([pose()]*10+[pose(d)]*duration+[pose()]*10);self.assertEqual(t.count,0)
 def test_observed_partial_motion_can_still_count(self):
  t,_=self.feed([pose()]*10+[pose(.2)]*5+[pose()]*5);self.assertEqual(t.count,1)
 def test_bottom_confirmation_not_return(self):
  t,_=self.feed([pose()]*10+[pose(.1)]*3+[pose()]*3)
  r=t.completion_receipts[0];self.assertEqual(r['bottom_confirmed_tick'],12);self.assertEqual(r['return_first_tick'],13);self.assertEqual(r['completed_tick'],15)
 def test_short_knee_gap_recovers_without_imputation(self):
  seq=[pose()]*10+[pose(.3)]*6+[pose(.3,0)]*4+[pose(.3)]*8+[pose()]*5
  raw=np.array(seq);t,segs=self.feed(seq);self.assertEqual(t.count,1);self.assertEqual(len(t.recovery_receipts),1)
  self.assertEqual(t.recovery_receipts[0]['observed_missing_slots'],4)
  self.assertTrue(np.isnan(raw[16:20,[25,26]]).all());self.assertGreaterEqual(t.last_completion_coverage,.8)
  self.assertFalse(any(k in [x[0]for x in segs[0]]for k in range(16,20)))
 def test_long_knee_gap_is_held(self):
  t,_=self.feed([pose()]*10+[pose(.3)]*6+[pose(.3,0)]*7+[pose(.3)]*8+[pose()]*5);self.assertEqual(t.count,0)
 def test_gap_does_not_supply_bottom_or_return_evidence(self):
  t,_=self.feed([pose()]*10+[pose(.3)]*2+[pose(.3,0)]*4+[pose()]*8);self.assertEqual(t.count,0)
 def test_core_gap_is_not_knee_only_recovery(self):
  p=pose(.3);p[23]=np.nan
  t,_=self.feed([pose()]*10+[pose(.3)]*6+[p]*4+[pose(.3)]*8+[pose()]*5);self.assertEqual(t.count,0)
 def test_one_knee_is_sufficient_but_raw_other_stays_missing(self):
  t,segs=self.feed([pose(0,1)]*10+[pose(.3,1)]*6+[pose(0,1)]*5);self.assertEqual(t.count,1);self.assertTrue(all(np.isnan(p[25]).all()for _,p in segs[0]))
 def test_initial_ready_cannot_be_assumed_without_knees(self):
  t,_=self.feed([pose(0,0)]*10+[pose(.3,0)]*6+[pose(0,0)]*5);self.assertEqual(t.count,0)
 def test_coverage_uses_actual_missing_slots(self):
  seq=[pose()]*10+[pose(.3)]*3+[pose(.3,0)]*6+[pose(.3)]*3+[pose()]*3
  t,_=self.feed(seq);self.assertEqual(t.count,0);self.assertLess(t.last_completion_coverage,.8)
class TimeAxisTest(unittest.TestCase):
 def base(self):
  t=MovementTracker('squats','front')
  for n in range(10):t.push(n,pose())
  return t
 def test_sparse_bottom_ticks_not_consecutive(self):
  t=self.base()
  for n in [10,12,14]:t.push(n,pose(.3))
  self.assertFalse(t.has_bottom);self.assertEqual(t.count,0)
 def test_sparse_return_ticks_not_consecutive(self):
  t=self.base()
  for n in [10,11,12]:t.push(n,pose(.3))
  for n in [13,15,17]:t.push(n,pose())
  self.assertEqual(t.returned,1);self.assertEqual(t.count,0)
 def test_three_missing_pushes_cannot_cover_seven_slots(self):
  t=self.base()
  for n in range(10,16):t.push(n,pose(.3))
  for n in [16,19,22]:t.push(n,pose(.3,0))
  self.assertFalse(t.active);self.assertFalse(t.knee_recovering);self.assertEqual(t.count,0)
 def test_sparse_recovery_ticks_not_consecutive(self):
  t=self.base()
  for n in range(10,16):t.push(n,pose(.3))
  t.push(16,pose(.3,0));t.push(17,pose(.3));t.push(19,pose(.3))
  self.assertTrue(t.knee_recovering);self.assertEqual(t.knee_recovery_streak,1)
 def test_one_recovery_tick_then_new_missing_does_not_split_long_gap(self):
  t=self.base()
  for n in range(10,16):t.push(n,pose(.3))
  for n in range(16,20):t.push(n,pose(.3,0))
  t.push(20,pose(.3));t.push(21,pose(.3,0));t.push(22,pose(.3,0))
  self.assertFalse(t.knee_recovering);self.assertFalse(t.active);self.assertEqual(t.count,0)
if __name__=='__main__':unittest.main()
