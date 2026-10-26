"""실제 저장본으로 순수 AI·규칙·혼합 점수와 입력 불변 계약 검사."""
import unittest,json
from pathlib import Path
import numpy as np
from front_candidate import vector,predict
class CandidateTest(unittest.TestCase):
 def test_saved_reps_are_scored_and_original_arrays_unchanged(self):
  p=Path(__file__).parent/'recordings/20261023-110200-964840';z=np.load(p/'pose.npz');xy=z['image_xy'][35:86];before=xy.copy();a=predict(xy)
  self.assertIsNone(a['pure_ai_total']);self.assertGreater(a['ai_depth_points'],0);self.assertEqual(a['maximum'],70)
  self.assertTrue(np.array_equal(before,xy,equal_nan=True))
  for key in ['ai_assisted_points','rule_points']:self.assertTrue(0<=a[key]<=70)
  noankles=xy.copy();noankles[:,27:]=np.nan;self.assertEqual(predict(noankles)['ai_depth_points'],a['ai_depth_points'])
 def test_no_person_is_not_scored(self):
  with self.assertRaises(ValueError):vector(np.full((40,33,2),np.nan))
 def test_training_does_not_include_local_video_or_validation_person(self):
  p=Path(__file__).parent/'front-candidate';rows=json.loads((p/'training-rows.json').read_text());t={r['person'] for r in rows if r['split']=='train'};v={r['person'] for r in rows if r['split']=='validation'}
  self.assertFalse(t&v);self.assertNotIn('P001',t)
