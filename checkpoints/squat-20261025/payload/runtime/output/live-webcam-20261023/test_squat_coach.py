"""지속 조건·누락 보류·발목 독립성을 검증하며 합성 자료는 학습에 쓰지 않는다."""
import unittest
import numpy as np
from squat_coach import LiveCoach,completed_feedback
class CoachTest(unittest.TestCase):
 def frame(self):
  x=np.full((33,2),np.nan)
  for j,p in [(11,[100,100]),(12,[200,100]),(23,[120,240]),(24,[180,240]),(25,[110,330]),(26,[190,330])]:x[j]=p
  return x
 def test_inward_spacing_requires_persistence(self):
  c=LiveCoach();x=self.frame();c.push(0,x,True);x[25,0]=140;x[26,0]=160
  for t in range(1,5):self.assertNotIn('무릎 간격',str(c.push(t,x,True)['messages']))
  self.assertIn('무릎 간격',str(c.push(5,x,True)['messages']))
 def test_missing_not_good_and_no_stale_warning(self):
  c=LiveCoach();x=self.frame();c.push(0,x,True);x[25:27]=np.nan;r=c.push(1,x,True)
  self.assertIn('보류',str(r['messages']))
  x[11:25]=np.nan;self.assertEqual(c.push(2,x,True)['status'],'관측 부족')
 def test_completed_reports_missing_knees_without_false_valgus(self):
  x=np.repeat(self.frame()[None],40,0);x[:,26]=np.nan;r=completed_feedback(x,None)
  self.assertIn('무릎 간격 변화 판단 보류',str(r['messages']))
  self.assertTrue(r['not_model_diagnosis'])

class RecordedFlowTest(unittest.TestCase):
 def test_five_saved_repetitions_reach_scoring_and_coaching(self):
  from pathlib import Path
  from movement import MovementTracker
  from front_candidate import predict
  z=np.load(Path(__file__).parent/'recordings/20261023-110200-964840/pose.npz');t=MovementTracker('squats','front');rows=[]
  for tick,xy in zip(z['frame_numbers'][:326],z['image_xy'][:326]):
   segment=t.push(int(tick),xy)
   if segment:
    points=np.array([p for _,p in segment]);c=predict(points);f=completed_feedback(points,c);rows.append((c,f))
  self.assertEqual(t.count,5);self.assertEqual(len(rows),5)
  self.assertTrue(all(c['maximum']==70 and f['messages'] for c,f in rows))
