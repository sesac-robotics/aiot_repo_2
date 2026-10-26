import unittest,json
from pathlib import Path
import numpy as np,joblib
from tracking_quality import features,assess
O=Path('/home/jayden/SeSACPAI-Mid/output/shoulder-single-audit-v1-20261025')
class TrackingContract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.rows=json.loads((O/'tracking-labels.frozen.json').read_text());cls.r=next(r for r in cls.rows if r['tracking_error']==0);cls.x=np.load(cls.r['cache'])['image_xy'][cls.r['cache_sample_index']].astype(float);cls.model=joblib.load(O/'tracking-quality-linear-v3.joblib')['model']
 def test_arms_head_and_legs_do_not_change_input(self):
  x=np.full((33,2),np.nan);x[[11,12,23,24]]=self.x[[11,12,23,24]];np.testing.assert_allclose(features(x),features(self.x))
 def test_rigid_body_tilt_is_not_a_quality_error_feature(self):
  t=.35;rotation=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]]);np.testing.assert_allclose(features(self.x@rotation),features(self.x),atol=1e-10)
 def test_translation_and_scale_invariance(self):np.testing.assert_allclose(features(self.x*1.5+[120,60]),features(self.x),atol=1e-10)
 def test_missing_hip_is_held_not_filled(self):
  x=self.x.copy();x[23]=np.nan;self.assertIsNone(features(x));self.assertEqual(assess(self.model,x)['status'],'held')
 def test_quality_model_never_outputs_grade(self):
  r=assess(self.model,self.x);self.assertIsNone(r['posture_grade']);self.assertIsNone(r['posture_score']);self.assertFalse(r['production_eligible'])
 def test_unvalidated_view_held(self):self.assertEqual(assess(self.model,self.x,'side')['status'],'held')
 def test_training_source_and_target_contract(self):
  self.assertEqual({r['split'] for r in self.rows},{'train'});self.assertTrue(all(r['label_target']=='four_joint_tracking_misconnection_not_movement_severity' for r in self.rows));people={}
  for r in self.rows:
   if r['source_sha256'] in people:self.assertEqual(people[r['source_sha256']],r['person'])
   people[r['source_sha256']]=r['person']
if __name__=='__main__':unittest.main()
