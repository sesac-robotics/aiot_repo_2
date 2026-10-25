import unittest
from pathlib import Path
import numpy as np
from rubric100 import assessment,ITEMS
class Rubric100Tests(unittest.TestCase):
 def test_original_weights_and_total(self):
  row={'items':[dict(id=k,weight=w,ai_grade=2,rule_grade=1) for k,t,w in ITEMS]}
  r=assessment(row);self.assertEqual([x['weight'] for x in r['items']],[30,20,30,20]);self.assertEqual(r['ai']['total'],100);self.assertEqual(r['rule']['total'],50)
 def test_missing_is_not_zero_or_renormalized(self):
  row={'items':[dict(id='trunk_balance',weight=20,ai_grade=1,rule_grade=2)]};r=assessment(row)
  self.assertIsNone(r['ai']['total']);self.assertEqual(r['ai']['earned_subtotal'],10);self.assertEqual(r['ai']['assessed_weight'],20);self.assertFalse(r['total_renormalized'])
 def test_zero_is_real_score_not_missing(self):
  r=assessment({'items':[dict(id='trunk_balance',weight=20,ai_grade=0,rule_grade=0)]});self.assertEqual(r['items'][3]['ai_points'],0)
 def test_knee_gap_and_height_return_do_not_fill_original_items(self):
  r=assessment({'accepted_items':{'items':[{'id':'knee_inward','grade':2}]},'front_candidate':{'return_height_grade':2,'trunk_grade':2}})
  self.assertIsNone(r['items'][1]['ai_points']);self.assertIsNone(r['items'][1]['rule_points']);self.assertIsNone(r['items'][2]['ai_points'])
 def test_single_normal_ai_not_claimed_validated_full(self):
  row={'items':[dict(id=k,weight=w,ai_grade=2,rule_grade=2,single_grade_training=(k=='return_extension')) for k,t,w in ITEMS]};r=assessment(row)
  self.assertIsNone(r['ai']['total']);self.assertEqual(r['ai']['assessed_weight'],80)
 def test_actual_scoring_and_no_feet(self):
  from server import score_samples
  z=np.load(Path(__file__).parent/'recordings/20261023-110200-964840/pose.npz');x=z['image_xy'][35:86].copy();x[:,27:]=np.nan
  # Anatomical core100 remains available unchanged; the frontal observation
  # contract has separately named knee-spacing and standing-height targets.
  r=score_samples(list(enumerate(x)),640,480,'squats','front',True)['legacy_score_100']
  self.assertEqual(sum(i['weight'] for i in r['items']),100);self.assertIsNotNone(r['items'][0]['ai_points']);self.assertIsNone(r['items'][2]['ai_points']);self.assertIsNone(r['ai']['total'])
