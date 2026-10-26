import unittest,sys,copy,json
from pathlib import Path
from shared_features import pool_item
B=Path(__file__).resolve().parents[2];sys.path.insert(0,str(B/'tools/error-enrichment'))
from split_guard import assert_training_allowed,POLICY
import numpy as np
from models import PositiveDeficitClassifier,MonotoneExtensionEnsemble
class Contracts(unittest.TestCase):
 def setUp(self):self.item=dict(item='return_extension',bank='front',eligible=True,feature=[140,145,150,155,160,165,170,.9],holds=[])
 def test_equivalent_banks_keep_measurement_and_source_immutable(self):
  for bank in ['front','side_legacy_roi_v1']:
   old=copy.deepcopy(self.item);old['bank']=bank;saved=copy.deepcopy(old);new=pool_item(old);self.assertEqual(new['feature'],old['feature']);new['feature'][0]=0;self.assertEqual(old,saved)
 def test_dimensionless_semantic_geometry_must_not_pool(self):
  for bank in ['side_semantic_roi_v1','side_semantic_roi_v1_aihub_transfer_v1','height_v4']:
   with self.assertRaises(ValueError):pool_item(dict(self.item,bank=bank))
 def test_missing_pose_cannot_become_score(self):
  out=pool_item(dict(self.item,eligible=False,feature=None,holds=['unobserved']));self.assertFalse(out['eligible']);self.assertIsNone(out['feature']);self.assertEqual(out['holds'],['unobserved'])
  with self.assertRaises(ValueError):pool_item(dict(self.item,feature=None))
 def test_malformed_or_rule_grade_features_rejected(self):
  for f in [[1,2,2],self.item['feature'][:-1]+[4],[float('nan')]*8,[180]*8,[170,165,160,155,150,145,140,1]]:
   with self.assertRaises(ValueError):pool_item(dict(self.item,feature=f))
 def test_wrong_anatomical_target_rejected(self):
  with self.assertRaises(ValueError):pool_item(dict(self.item,item='standing_height_return'))
 def test_Test_source_cannot_be_renamed_into_Train(self):
  policy=json.loads(POLICY.read_text());source=policy['protected_source_sha256'][0]
  with self.assertRaises(ValueError):assert_training_allowed([dict(id='new_alias',person='new_person',split='train',sources=[source])])
 def test_development_cannot_fit(self):
  with self.assertRaises(ValueError):assert_training_allowed([dict(id='x',person='new_person',split='validation',sources=['unused'])])
class MonotonicSeverity(unittest.TestCase):
 def setUp(self):
  x=np.array([[v]*7+[1.] for v in [100,120,140,165,175,179]],float);y=[1,1,1,2,2,2]
  model=PositiveDeficitClassifier(1).fit(MonotoneExtensionEnsemble.deficits(x),y)
  self.model=MonotoneExtensionEnsemble([model])
 def test_more_extension_never_increases_error_probability(self):
  x=np.array([[v]*7+[1.] for v in range(90,181)]);p=self.model.normal_probability(x);self.assertTrue(np.all(np.diff(p)>=-1e-12))
 def test_visibility_is_not_a_severity_shortcut(self):
  x=np.array([[160.]*7+[v] for v in [.8,.9,1.]]);p=self.model.normal_probability(x);np.testing.assert_allclose(p,p[0],atol=1e-12)
 def test_untrained_zero_grade_never_returned(self):
  values=self.model.predict([[0.]*7+[1.],[180.]*7+[1.]]);self.assertTrue(set(values)<={1.,2.})
 def test_wrong_units_rejected(self):
  for x in [[[181.]*7+[1.]],[[float('nan')]*8],[[90.]*7]]:
   with self.assertRaises(ValueError):self.model.predict(x)
if __name__=='__main__':unittest.main()
