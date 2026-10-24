import sys,unittest
from pathlib import Path
import numpy as np
APP=Path('/home/jayden/SeSACPAI-Mid/output/live-webcam-20261023');sys.path.insert(0,str(APP))
from movement_candidate import MovementTracker
from test_movement import squat

def pose(drop):
 p=squat();p[[11,12,23,24],1]+=100*drop;return p
class TransitionTests(unittest.TestCase):
 def base(self):
  t=MovementTracker('squats','front')
  for n in range(10):t.push(n,pose(0))
  return t
 def test_overlap_does_not_count_as_return(self):
  for value in [.085,.089]:
   with self.subTest(value=value):
    t=self.base();t.push(10,pose(.095));t.push(11,pose(value))
    self.assertTrue(t.has_bottom);self.assertEqual(t.bottom_confirmed_tick,11)
    self.assertEqual(t.returned,0);self.assertIsNone(t.return_first_tick)
    t.push(12,pose(0));t.push(13,pose(0));self.assertEqual(t.count,0)
    t.push(14,pose(0));self.assertEqual(t.count,1)
    self.assertEqual(t.completion_receipts,[dict(number=1,bottom_confirmed_tick=11,return_first_tick=12,completed_tick=14)])
 def test_bottom_plateau_is_not_return(self):
  t=self.base()
  for n in range(10,20):t.push(n,pose(.095))
  self.assertTrue(t.has_bottom);self.assertEqual(t.returned,0);self.assertEqual(t.count,0)
 def test_broken_return_streak_records_new_first_tick(self):
  t=self.base();t.push(10,pose(.1));t.push(11,pose(.1));t.push(12,pose(0));t.push(13,pose(.085))
  self.assertEqual(t.returned,0);self.assertIsNone(t.return_first_tick)
  for n in [14,15,16]:t.push(n,pose(0))
  self.assertEqual(t.completion_receipts[0]['return_first_tick'],14)
 def test_observation_loss_clears_unfinished_receipts(self):
  t=self.base();t.push(10,pose(.1));t.push(11,pose(.1));t.push(12,pose(0))
  for n in range(13,17):t.push(n,np.full((33,2),np.nan))
  self.assertIsNone(t.bottom_confirmed_tick);self.assertIsNone(t.return_first_tick);self.assertEqual(t.completion_receipts,[])
 def test_thresholds_stay_unchanged(self):
  t=self.base()
  for n in range(10,20):t.push(n,pose(.079))
  self.assertFalse(t.has_bottom);self.assertEqual(t.count,0)
if __name__=='__main__':unittest.main()
