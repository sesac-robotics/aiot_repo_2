"""단일 등급 학습 결과를 검증된 만점으로 표시하지 않는지 검사한다."""
import unittest
from server import observed_points


class DisplayTest(unittest.TestCase):
    def test_single_grade_is_not_a_validated_score(self):
        result={'items':[dict(id='return_extension',weight=20,ai_grade=2.,rule_grade=None,single_grade_training=True)]}
        shown=observed_points(result,'ai')
        self.assertEqual(shown['maximum'],0)
        self.assertFalse(shown['full_score'])
        self.assertEqual(shown['excluded_unvalidated_items'],['return_extension'])

    def test_missing_item_does_not_become_full_score(self):
        result={'items':[dict(id='depth',weight=30,ai_grade=1.,rule_grade=1,single_grade_training=False),dict(id='alignment',weight=70,ai_grade=None,rule_grade=None,single_grade_training=False)]}
        shown=observed_points(result,'ai')
        self.assertEqual(shown['earned'],15.)
        self.assertEqual(shown['maximum'],30)
        self.assertFalse(shown['full_score'])

if __name__=='__main__':unittest.main()

class FullScoringAdapterTest(unittest.TestCase):
    def test_saved_segment_uses_real_engine_and_returns_candidate_and_feedback(self):
        import numpy as np
        from pathlib import Path
        from server import score_samples
        z=np.load(Path(__file__).parent/'recordings/20261023-110200-964840/pose.npz')
        samples=[(int(t),p) for t,p in zip(z['frame_numbers'][35:86],z['image_xy'][35:86])]
        r=score_samples(samples,640,480,'squats','front',True)
        self.assertIn('front_candidate',r)
        self.assertFalse(r['front_candidate'].get('error'))
        self.assertTrue(r['coaching']['messages'])
