"""정면 정책의 원본 보존·범위 제한·미관측 처리 회귀 검사."""
import unittest
import numpy as np
from front_policy import apply_front_policy,frontal_motion_observations

class FrontPolicyTest(unittest.TestCase):
 def result(self):
  return dict(ai_score=None,rule_score=None,items=[dict(id=k,ai_grade=2.,rule_grade=2,ai_minus_rule=0.,single_grade_training=(k=='trunk_balance'),reasons=[]) for k in ['descent_range','return_extension','trunk_balance']])
 def test_readonly_and_target_scope(self):
  source=self.result();out=apply_front_policy(source,'squats','front')
  self.assertEqual(source['items'][0]['ai_grade'],2.)
  self.assertIsNone(out['items'][0]['ai_grade'])
  self.assertIsNone(out['items'][2]['ai_grade'])
  self.assertEqual(out['items'][2]['rule_grade'],2)
  for exercise,view in [('pushups','front'),('squats','side'),('squats','front_oblique')]:self.assertEqual(apply_front_policy(source,exercise,view),source)
 def test_reapply_preserves_original_audit(self):
  a=apply_front_policy(self.result(),'squats','front');b=apply_front_policy(a,'squats','front')
  self.assertEqual(a,b)
 def test_no_ankles_needed_and_missing_not_filled(self):
  xy=np.full((20,33,2),np.nan)
  for j,point in [(11,[0,0]),(12,[10,0]),(23,[0,10]),(24,[10,10]),(25,[0,20]),(26,[10,20])]:xy[:,j]=point
  r=frontal_motion_observations(xy);self.assertEqual(r['status'],'observed_proxy_only');self.assertIsNone(r['grade'])
  xy[:,23]=np.nan;self.assertEqual(frontal_motion_observations(xy)['status'],'insufficient_observation')
