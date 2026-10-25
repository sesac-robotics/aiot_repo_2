import unittest
from pathlib import Path
import numpy as np
from front_candidate import predict
from method_comparison import compare_candidate
class ComparisonTests(unittest.TestCase):
 def candidate(self):
  z=np.load(Path(__file__).parent/'recordings/20261023-110200-964840/pose.npz');return predict(z['image_xy'][35:86])
 def test_explanation_exactly_reconstructs_actual_model_and_rule(self):
  c=self.candidate();r=compare_candidate(c);m=r['model_evidence'];rule=r['rule_evidence'];low,high=rule['boundaries'];v=rule['value']
  self.assertAlmostEqual(m['baseline']+sum(x['contribution'] for x in m['contributions']),m['prediction'],places=6)
  self.assertEqual(r['rule_grade'],2 if v>=high else 1 if v>=low else 0)
  self.assertEqual(r['ai_points'],c['ai_depth_points']);self.assertEqual(r['maximum'],30)
  self.assertFalse(r['gpt_used']);self.assertFalse(r['whole_posture_total'])
 def test_mismatched_model_output_is_never_explained_as_same_source(self):
  c=self.candidate();c['ai_depth_grade']+=.5
  self.assertEqual(compare_candidate(c)['status'],'source_mismatch')
 def test_unobservable_input_does_not_generate_explanation(self):
  self.assertEqual(compare_candidate({'error':'missing'})['status'],'unavailable')
 def test_real_scoring_path_contains_comparison_without_ankles(self):
  from server import score_samples
  z=np.load(Path(__file__).parent/'recordings/20261023-110200-964840/pose.npz');x=z['image_xy'][35:86].copy();x[:,27:]=np.nan
  r=score_samples([(n,a) for n,a in enumerate(x)],640,480,'squats','front',True)
  self.assertIn(r['method_comparison']['status'],['different','close']);self.assertIsNotNone(r['method_comparison']['ai_points'])
