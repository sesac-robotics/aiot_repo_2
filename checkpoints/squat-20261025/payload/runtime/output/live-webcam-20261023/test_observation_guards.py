import unittest
import numpy as np
from accepted_items import evaluate
from test_movement import squat
from movement import MovementTracker
class GuardTests(unittest.TestCase):
 def test_no_count_limit_and_shallow_return_closes(self):
  t=MovementTracker('squats','front');stand=squat();small=stand.copy();small[[11,12,23,24],1]+=20
  seq=[stand]*10
  for _ in range(7):seq += [small]*8+[stand]*8
  segments=[]
  for tick,x in enumerate(seq):
   s=t.push(tick,x)
   if s:segments.append(s)
  self.assertEqual(t.count,7)
  self.assertTrue(all(s[-1][0]-s[0][0]<25 for s in segments))
 def test_small_noise_is_not_repetition(self):
  t=MovementTracker('squats','front');stand=squat()
  for tick in range(100):
   x=stand.copy();x[[11,12,23,24],1]+=2*np.sin(tick)
   t.push(tick,x)
  self.assertEqual(t.count,0)
 def test_border_feet_are_not_scored_as_supported(self):
  from pathlib import Path
  z=np.load(Path(__file__).parent.parent/'squat-relabel-expanded-20261023/observations/Mima_01/front.npz');x=z['image_xy'].copy()
  height=1920;width=1080;x[:,[29,30,31,32],1]=height+1
  r=evaluate(x,width,height);body,foot=r['items']
  self.assertIsNotNone(body['grade']);self.assertIsNone(foot['grade']);self.assertEqual(foot['reason'],'frame_edge')
 def test_sustained_small_attempt_not_merged_into_next(self):
  t=MovementTracker('squats','front');stand=squat();tiny=stand.copy();tiny[[11,12,23,24],1]+=5
  seq=[stand]*10+[tiny]*10+[stand]*30+[squat(True)]*10+[stand]*10
  segments=[]
  for tick,x in enumerate(seq):
   r=t.push(tick,x)
   if r:segments.append(r)
  self.assertEqual(t.count,1);self.assertLess(segments[0][-1][0]-segments[0][0][0],30)

class PreviousSessionRegression(unittest.TestCase):
 def test_previously_reviewed_five_and_three_sessions(self):
  from pathlib import Path
  root=Path(__file__).parent
  for folder,expected in [('recordings/20261023-110200-964840',5),('sessions/1792722400309382306',3)]:
   with self.subTest(folder=folder):
    z=np.load(root/folder/'pose.npz');t=MovementTracker('squats','front')
    for tick,xy in zip(z['frame_numbers'],z['image_xy']):t.push(int(tick),xy)
    self.assertEqual(t.count,expected)
