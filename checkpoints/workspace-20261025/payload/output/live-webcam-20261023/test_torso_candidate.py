"""몸통 입력 독립성·강건 복귀·기존 오류 사례의 후보 동작 회귀 검사."""
import json,unittest
from pathlib import Path
import numpy as np
import joblib
from torso_candidate import torso_features
from front_candidate import predict
R=Path(__file__).parent
class TorsoTest(unittest.TestCase):
 def test_no_knees_or_ankles_required_for_torso(self):
  z=np.load(R/'recordings/20261023-110200-964840/pose.npz');xy=z['image_xy'][140:196].copy();a=torso_features(xy);xy[:,25:]=np.nan
  np.testing.assert_allclose(a,torso_features(xy))
 def test_shifted_rep3_return_not_false_failure(self):
  z=np.load(R/'recordings/20261023-110200-964840/pose.npz')
  for a,b in [(140,195),(135,190),(145,200)]:self.assertEqual(predict(z['image_xy'][a:b+1])['return_height_grade'],2)
 def test_one_known_training_error_recognized_not_generalization(self):
  rows=json.loads((R/'front-candidate/torso-training-rows.json').read_text());bad=next(r for r in rows if r['trunk_grade']==0);m=joblib.load(R/'front-candidate/torso-model.joblib')
  self.assertLess(m['body_model'].predict([bad['features'][1:]])[0],1)
