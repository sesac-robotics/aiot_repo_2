import sys,unittest,importlib.util
from pathlib import Path
import numpy as np
import features as fixed
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/paired-evaluation'))
from thresholds import allowed_indices
B=Path(__file__).resolve().parents[2];spec=importlib.util.spec_from_file_location('dynamic_scale',B/'tools/return-discrimination/features.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
class FixedReference(unittest.TestCase):
 def setUp(self):
  self.x=np.full((12,33,2),np.nan);self.ref=np.full((3,33,2),np.nan)
  for k in [0,1]:self.ref[:,[11+k,23+k,25+k,27+k]]=[[[100+20*k,100],[100+20*k,200],[100+20*k,300],[100+20*k,400]]]
  self.x[:]=self.ref[0];self.base=dict(return_index=6,bottom_index=2)
 def test_ready_zero(self):np.testing.assert_allclose(fixed.extract(self.x,self.ref,self.base)['x'],0)
 def test_global_scale_and_translation_do_not_change_deficit(self):
  self.x[6:9,[11,12,23,24],1]+=10;a=fixed.extract(self.x,self.ref,self.base)['x'];b=fixed.extract(self.x*2+15,self.ref*2+15,self.base)['x'];np.testing.assert_allclose(a,b)
 def test_return_scale_cannot_cancel_height_loss(self):
  for k in [0,1]:self.x[6:9,[11+k,23+k,25+k,27+k]]=[[[100+20*k,115],[100+20*k,210],[100+20*k,305],[100+20*k,400]]]
  a=fixed.extract(self.x,self.ref,self.base)['x'];b=old.extract(self.x,self.ref,self.base)['x'];self.assertGreater(a[0],.04);self.assertAlmostEqual(b[0],0,places=6)
 def test_support_translation_is_removed(self):
  self.x[6:9]+=20;np.testing.assert_allclose(fixed.extract(self.x,self.ref,self.base)['x'],0)
 def test_missing_reference_is_not_filled(self):self.assertIsNone(fixed.extract(self.x,self.ref[:2],self.base))
 def test_no_completed_return_abstains(self):self.assertIsNone(fixed.extract(self.x,self.ref,None))
 def test_missing_both_supports_abstains(self):
  self.x[:,[27,28]]=np.nan;self.assertIsNone(fixed.extract(self.x,self.ref,self.base))
 def test_large_view_scale_change_is_held(self):
  self.x[6:9]*=.5;self.assertIsNone(fixed.extract(self.x,self.ref,self.base))
 def test_auxiliary_angles_cannot_be_used_by_height_rule(self):self.assertEqual(allowed_indices('standing_fixed_initial_context_v3','standing_height_return',10),[0,1,2,3])
 def test_height_only_rule_contract(self):self.assertEqual(allowed_indices('standing_fixed_initial_height_v2','standing_height_return',4),[0,1,2,3])
if __name__=='__main__':unittest.main()
