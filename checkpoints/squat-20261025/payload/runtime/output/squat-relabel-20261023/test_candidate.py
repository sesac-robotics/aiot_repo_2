import unittest
import json
from pathlib import Path
import numpy as np
from criteria_features import observed_features
from predict import predict
P=Path(__file__).parent
class CandidateSafety(unittest.TestCase):
 def test_missing_feet_never_become_good(self):
  x=np.load(P/'observations/Mima_01/front.npz')['image_xy'].copy()
  x[:,[27,28,29,30,31,32]]=np.nan
  p=predict(x)
  self.assertEqual(p['items']['heel_support']['status'],'unobserved')
  self.assertEqual(p['items']['knee_inward']['status'],'experimental_only')
  self.assertIsNone(p['total'])
  self.assertFalse(p['promoted'])
 def test_translation_scale_invariance(self):
  x=np.load(P/'observations/Mima_01/front.npz')['image_xy']
  a,b=observed_features(x),observed_features(x*1.5+[150,200])
  for k in a:np.testing.assert_allclose(a[k]['x'],b[k]['x'],rtol=1e-5,atol=1e-5)
 def test_people_disjoint(self):
  rows=json.loads((P/'feature-rows.json').read_text())
  train={r['person'] for r in rows if r['split']=='train'}
  val={r['person'] for r in rows if r['split']=='validation'}
  self.assertFalse(train&val)
if __name__=='__main__':unittest.main()
