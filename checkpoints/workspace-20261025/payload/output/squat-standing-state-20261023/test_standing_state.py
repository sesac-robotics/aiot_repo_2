import sys,json,unittest
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;A=O.parent/'live-webcam-20261023';sys.path.insert(0,str(A))
from standing_state import make_reference,features,predict_frame,evaluate_return
class StandingStateTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.z=np.load(O.parent/'squat-return-20261023/squat_front_m1_9_4-heavy.npz');cls.x=cls.z['image_xy'];cls.f=cls.z['frame_numbers'];cls.ref=make_reference(np.load(O/'squat_front_m1_9_4-reference.npz')['image_xy']);cls.seg=cls.x[(cls.f>=485)&(cls.f<=530)]
 def test_standing_alone_is_not_a_return(self):
  x=np.repeat(self.x[2:3],20,axis=0);self.assertEqual(evaluate_return(x,make_reference(x))['status'],'no_descent')
 def test_active_descent_does_not_get_a_return_score(self):self.assertIsNone(evaluate_return(self.seg,self.ref,phase='active')['ai_grade'])
 def test_missing_end_is_not_filled_as_standing(self):
  x=self.seg.copy();x[-3:,[11,12,23,24]]=np.nan;self.assertIsNone(evaluate_return(x,self.ref)['ai_grade'])
 def test_head_ankles_and_knees_do_not_gate_core_state(self):
  x=self.seg[-1].copy();a=predict_frame(x,self.ref);x[:11]=np.nan;x[25:]=np.nan;b=predict_frame(x,self.ref);self.assertIsNotNone(b);self.assertEqual(a['standing_probability'],b['standing_probability'])
 def test_global_pixel_translation_and_scaling_invariance(self):
  x=self.seg[-1];a=features(x,self.ref)['core'];r=dict(points=self.ref['points']*2+[50,80],scale=self.ref['scale']*2);b=features(x*2+[50,80],r)['core'];np.testing.assert_allclose(a,b,atol=1e-8)
 def test_observed_partial_end_is_not_a_confirmed_return(self):
  r=evaluate_return(self.seg,self.ref);self.assertFalse(r['standing_confirmed']);self.assertGreaterEqual(r['ai_points'],0);self.assertLessEqual(r['ai_points'],20);self.assertFalse(r['rule_used_in_ai_score'])
 def test_people_and_sources_are_disjoint(self):
  d=json.loads((O/'feature-rows.json').read_text());tr=[r for r in d if r['split']=='train'];te=[r for r in d if r['split']=='validation'];self.assertFalse({r['person'] for r in tr}&{r['person'] for r in te});self.assertFalse({r['source'] for r in tr}&{r['source'] for r in te})
 def test_old_saved_rows_remain_unscored_without_state_model_evidence(self):
  from unified_assessment import apply
  d=json.loads((A/'sessions/1792744784509293036/final-state.json').read_text());r=apply(d['repetitions'][0]);self.assertIsNone(r['score_100']['items'][1]['ai_points'])
 def test_same_snapshot_drives_points_and_guidance(self):
  if not (O/'replay-state.json').exists():self.skipTest('replay build pending')
  d=json.loads((O/'replay-state.json').read_text());self.assertEqual(len(d['repetitions']),8)
  for r in d['repetitions']:
   i=r['score_100']['items'][1];self.assertIsNotNone(i['ai_points']);self.assertEqual(i['ai_points'],i['evidence']['standing_state']['ai_points']);self.assertTrue(i['evidence']['standing_state_model']);self.assertTrue(i['evidence']['standing_confirmed']);self.assertEqual(r['ai']['maximum'],100)
 def test_return_followup_preserves_other_item_scores(self):
  old=json.loads((A/'sessions/1792744784509293036/final-state.json').read_text())
  new=json.loads((O/'replay-state.json').read_text())
  for before,after in zip(old['repetitions'],new['repetitions']):
   for a,b in zip(before['score_100']['items'],after['score_100']['items']):
    if a['id']=='standing_return':continue
    self.assertEqual(a['ai_points'],b['ai_points'])
    self.assertEqual(a['rule_points'],b['rule_points'])
if __name__=='__main__':unittest.main()
