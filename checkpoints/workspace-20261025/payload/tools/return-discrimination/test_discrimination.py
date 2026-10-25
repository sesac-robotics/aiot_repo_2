import json,sys,unittest,pickle,importlib.util
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[2];sys.path[:0]=[str(B/'tools/return-discrimination'),str(B/'tools/error-enrichment'),str(B/'tools/paired-evaluation')]
import features
from models import PositiveDeficitClassifier,GroupCalibratedDeficitEnsemble
from thresholds import allowed_indices
class GeometryTests(unittest.TestCase):
 def setUp(self):
  self.x=np.full((12,33,2),np.nan)
  for side in [0,1]:self.x[:,[11+side,23+side,25+side,27+side],:]=[[[100+20*side,100],[100+20*side,200],[100+20*side,300],[100+20*side,400]]]
  self.ref=self.x[:3].copy();self.base={'return_index':6,'bottom_index':2}
 def test_ready_zero(self):self.assertTrue(np.allclose(features.extract(self.x,self.ref,self.base)['x'],0))
 def test_scale_and_translation_invariance(self):
  a=features.extract(self.x,self.ref,self.base)['x'];b=features.extract(self.x*2+35,self.ref*2+35,self.base)['x'];self.assertTrue(np.allclose(a,b))
 def test_missing_both_supports_abstains(self):
  self.x[:,:,0]=np.nan;self.assertIsNone(features.extract(self.x,self.ref,self.base))
 def test_no_imputation_of_reference(self):self.assertIsNone(features.extract(self.x,self.ref[:2],self.base))
 def test_lower_shoulders_raise_height_deficit(self):
  self.x[6:9,[11,12],1]+=25;v=features.extract(self.x,self.ref,self.base)['x'];self.assertGreater(v[2],0);self.assertAlmostEqual(v[8],0)
 def test_side_exchange_invariance(self):
  self.x[6:9,11,1]+=20;a=features.extract(self.x,self.ref,self.base)['x'];x=self.x.copy();r=self.ref.copy()
  for l,q in [(11,12),(23,24),(25,26),(27,28)]:x[:,[l,q]]=x[:,[q,l]];r[:,[l,q]]=r[:,[q,l]]
  self.assertTrue(np.allclose(a,features.extract(x,r,self.base)['x']))
 def test_height_rules_cannot_use_angle_only(self):self.assertEqual(allowed_indices('standing_support_extension_v1','standing_height_return',10),[0,1,2,3])
class ModelTests(unittest.TestCase):
 def setUp(self):self.m=PositiveDeficitClassifier(1).fit([[0,0],[.01,.02],[.2,.3],[.3,.4]],[2,2,1,1])
 def test_positive_coefficients(self):self.assertTrue((self.m.coef_>=0).all())
 def test_error_risk_increases(self):self.assertGreater(self.m.normal_probability([[0,0]])[0],self.m.normal_probability([[.4,.5]])[0])
 def test_ensemble_serialization(self):
  a=GroupCalibratedDeficitEnsemble([self.m,self.m],.6);self.assertTrue(np.allclose(a.normal_probability([[.1,.2]]),pickle.loads(pickle.dumps(a)).normal_probability([[.1,.2]])))
 def test_no_fabricated_zero_grade(self):self.assertEqual(set(self.m.predict([[0,0],[.3,.4]])),{1,2})
 def test_nested_train_members(self):
  O=B/'output/return-discrimination-ensemble-20261024';h=json.loads((O/'new-head-training.json').read_text());rs={r['id']:r for r in json.loads((O/'score-inputs.frozen.json').read_text())}
  for fold in h['nested_CV_folds']:
   self.assertFalse(set(fold['fit_groups'])&set(fold['held_groups']))
   for member in fold['inner_members']:
    self.assertFalse(set(member['fit_groups'])&set(member['held_groups']));self.assertFalse(set(member['fit_ids'])&set(member['held_ids']));self.assertTrue(set(member['fit_ids']+member['held_ids'])<=set(fold['fit_ids']));self.assertFalse(set(member['fit_ids'])&set(fold['outer_ids']))
  for m in h['component_training']:
   self.assertTrue(all(rs[i]['split']=='train' for i in m['fit_ids']));self.assertFalse({s for i in m['fit_ids'] for s in rs[i]['sources']}&{s for i in m['held_ids'] for s in rs[i]['sources']})
if __name__=='__main__':unittest.main()
