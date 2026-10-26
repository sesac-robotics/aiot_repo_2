import json,hashlib,unittest
from pathlib import Path
import numpy as np,joblib
from features import extract,reference
from predict import predict
O=Path(__file__).resolve().parent
class ReturnTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.z=np.load(O/'squat_front_m1_9_4-heavy.npz');cls.x=cls.z['image_xy'];cls.frames=cls.z['frame_numbers'];cls.ref=reference(cls.x[(cls.frames>=51)&(cls.frames<=59)]);cls.segment=cls.x[(cls.frames>=485)&(cls.frames<=530)]
 def test_head_and_ankles_are_not_required(self):
  a=extract(self.segment,self.ref);x=self.segment.copy();x[:,:11]=np.nan;x[:,27:]=np.nan;b=extract(x,self.ref);self.assertIsNotNone(a);np.testing.assert_allclose(a['x'],b['x'])
 def test_missing_end_is_unobserved_not_normal(self):
  x=self.segment.copy();x[-3:,[11,12,23,24]]=np.nan;self.assertIsNone(extract(x,self.ref))
 def test_stationary_pose_is_not_a_return(self):
  x=np.repeat(self.x[18:19],20,axis=0);self.assertIsNone(extract(x,reference(x)))
 def test_pixel_scale_and_translation_do_not_change_score_features(self):
  a=extract(self.segment,self.ref);x=self.segment*1.5+[30,50];r=dict(self.ref,hip_y=self.ref['hip_y']*1.5+50,shoulder_y=self.ref['shoulder_y']*1.5+50,center_x=self.ref['center_x']*1.5+30,scale=self.ref['scale']*1.5,torso_x=self.ref['torso_x']*1.5);b=extract(x,r);np.testing.assert_allclose(a['x'],b['x'],atol=1e-8)
 def test_failed_validation_cannot_become_runtime_score(self):
  p=predict(self.segment,self.ref);self.assertIsNone(p['ai_points']);self.assertEqual(p['status'],'validation_pending')
 def test_candidate_is_actual_learned_prediction_with_twenty_point_weight(self):
  p=predict(self.segment,self.ref,allow_unapproved=True);b=joblib.load(O/'candidate.joblib');self.assertEqual(p['ai_grade'],float(b['model'].predict([p['features']['x']])[0]));self.assertEqual(p['ai_points'],round(p['ai_grade']*10,1));self.assertFalse(p['rule_used_in_ai_score']);self.assertFalse(p['gpt_used_in_inference'])
 def test_label_and_person_provenance(self):
  b=joblib.load(O/'candidate.joblib');self.assertEqual(b['labels_sha256'],hashlib.sha256((O/'labels.frozen.json').read_bytes()).hexdigest());rows=json.loads((O/'feature-rows.json').read_text());a=[r for r in rows if r['split']=='train'];v=[r for r in rows if r['split']=='validation'];self.assertFalse({r['person'] for r in a}&{r['person'] for r in v});self.assertFalse({r['source'] for r in a}&{r['source'] for r in v});self.assertNotIn('cfrep_w4',b['training_people'])
 def test_grade_zero_is_not_claimed_as_trained(self):
  b=joblib.load(O/'candidate.joblib');self.assertEqual(b['supported_grades'],[1,2]);self.assertNotIn('0',b['train_support'])
if __name__=='__main__':unittest.main()
