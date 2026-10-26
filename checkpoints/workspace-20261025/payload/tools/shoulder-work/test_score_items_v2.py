import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'output/live-webcam-20261023'))
from test_shoulder_candidate import pose
from score_items_v2 import score,VERSION,TARGETS
class Dummy:
 def __init__(self,value):self.value=value
 def predict(self,x):return np.full(len(x),self.value)
class Tests(unittest.TestCase):
 def samples(self):return list(enumerate([pose()]*6+[pose(True)]*4+[pose()]*3))
 def model(self,value=1.5):return dict(model=Dummy(value),training_grade_counts={1:3,2:6},feature_schema=VERSION,criterion=TARGETS['lower_return'],view='front',development_gate={'passed':True},training_ids=['example'],training_source_hashes=['example'])
 def test_no_model_no_fabricated_points(self):self.assertTrue(all(r['points'] is None for r in score(self.samples())['items']))
 def test_normal_only_blocked(self):
  m=self.model();m['training_grade_counts']={2:100};r=score(self.samples(),{'lower_return':m});self.assertIsNone(r['items'][0]['points'])
 def test_wrong_feature_blocked(self):
  m=self.model();m['feature_schema']='old';self.assertIsNone(score(self.samples(),{'lower_return':m})['items'][0]['points'])
 def test_gate_failed_blocked(self):
  m=self.model();m['development_gate']={'passed':False};self.assertIsNone(score(self.samples(),{'lower_return':m})['items'][0]['points'])
 def test_severe_unseen_not_extrapolated(self):self.assertIsNone(score(self.samples(),{'lower_return':self.model(0)})['items'][0]['points'])
 def test_partial_supported_example_only(self):
  r=score(self.samples(),{'lower_return':self.model()});self.assertEqual(r['items'][0]['points'],18.75);self.assertIsNone(r['items'][1]['points']);self.assertFalse(r['all_items_supported'])
 def test_reference_and_criterion_not_interchangeable(self):
  self.assertIsNone(score(self.samples(),{'frontal_trunk_hold':self.model()})['items'][1]['points'])
if __name__=='__main__':unittest.main()
