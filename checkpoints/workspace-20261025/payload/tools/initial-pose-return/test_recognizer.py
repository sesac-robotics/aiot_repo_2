import unittest,numpy as np
from recognizer import ReturnTracker,pose,relative
class ReadyReturnTests(unittest.TestCase):
 def feed(self,t,values,start=0):
  for k,p in enumerate(values):r=t.update(start+k*.1,p)
  return r
 def test_staying_ready_is_not_a_repetition(self):
  r=self.feed(ReturnTracker('pushups'),[.95]*20);self.assertEqual(r['return_events'],0)
 def test_initial_motion_without_ready_does_not_become_return(self):
  r=self.feed(ReturnTracker('pushups'),[.01]*5+[.95]*5);self.assertEqual(r['return_events'],0)
 def test_departure_then_stable_return_is_observed_once(self):
  r=self.feed(ReturnTracker('pushups'),[.95]*3+[.01]*4+[.95]*8);self.assertEqual(r['return_events'],1);self.assertIsNone(r['grade']);self.assertIsNone(r['valid_repetition'])
 def test_intermediate_flexed_pose_is_not_failed_repetition(self):
  t=ReturnTracker('squats');r=self.feed(t,[.95]*3+[.01]*5);self.assertEqual(r['status'],'movement_away_from_initial');self.assertIsNone(r['grade']);self.assertEqual(t.finish()['status'],'observation_ended')
 def test_user_stop_never_becomes_a_failure_grade(self):
  t=ReturnTracker('shoulder');self.feed(t,[.95]*3+[.01]*3);r=t.finish(attempt_closed=True,user_stopped=True);self.assertEqual(r['status'],'stopped_by_user');self.assertIsNone(r['grade'])
 def test_missing_observation_does_not_prove_a_completed_cycle(self):
  t=ReturnTracker('ytw');r=self.feed(t,[.95]*3+[.01]*3+[None]+[.95]*3);self.assertEqual(r['status'],'return_observed_cycle_unverified');self.assertEqual(r['return_events'],0)
 def test_hold_exercise_does_not_create_repetition_return_counts(self):
  t=ReturnTracker('plank');r=self.feed(t,[.95]*3+[.01]*3+[.95]*3);self.assertEqual(r['return_events'],0);self.assertEqual(r['status'],'holding')
 def test_single_noisy_ready_frame_is_not_a_return(self):
  r=self.feed(ReturnTracker('pushups'),[.95]*3+[.01]*4+[.95,.01,.01]);self.assertEqual(r['return_events'],0)
 def test_clock_gap_cannot_bridge_a_cycle(self):
  t=ReturnTracker('pushups');self.feed(t,[.95]*3+[.01]*3);t.update(2.,.95);r=self.feed(t,[.95]*3,2.1);self.assertEqual(r['return_events'],0)
 def test_unobserved_required_joint_is_not_filled(self):
  x=np.ones((33,2));x[11]=[2,2];x[12]=[3,2];x[23]=[2,5];x[24]=[3,5];x[15]=np.nan;self.assertIsNone(pose('pushups',x))
 def test_relative_features_reject_shape_and_missing_reference(self):
  with self.assertRaises(ValueError):relative([1,2],[1,np.nan])
  with self.assertRaises(ValueError):relative([1,2],[1])

class PhaseAndInferenceTests(unittest.TestCase):
 def test_Y_T_W_completes_without_forced_arms_down(self):
  from recognizer import YTWPhaseTracker
  t=YTWPhaseTracker()
  for i,p in enumerate(['Y']*3+['T']*3+['W']*5):r=t.update(i*.1,{c:float(c==p) for c in ['Y','T','W']})
  self.assertEqual(r['ordered_cycles'],1);self.assertIsNone(r['grade'])
 def test_skipping_T_does_not_complete_YTW(self):
  from recognizer import YTWPhaseTracker
  t=YTWPhaseTracker()
  for i,p in enumerate(['Y']*3+['W']*3):r=t.update(i*.1,{c:float(c==p) for c in ['Y','T','W']})
  self.assertEqual(r['ordered_cycles'],0)
 def test_phase_gap_cannot_complete_order(self):
  from recognizer import YTWPhaseTracker
  t=YTWPhaseTracker()
  for i,p in enumerate(['Y']*3+['T']*3):t.update(i*.1,{c:float(c==p) for c in ['Y','T','W']})
  t.update(.6,None)
  for i in range(3):r=t.update(.7+i*.1,{'Y':0.,'T':0.,'W':1.})
  self.assertEqual(r['ordered_cycles'],0)
 def test_visible_side_is_used_without_inventing_occluded_arm(self):
  x=np.full((33,2),np.nan);x[11]=[100,100];x[13]=[100,150];x[15]=[100,200];x[23]=[200,100]
  self.assertIsNotNone(pose('pushups',x));x[15]=np.nan;self.assertIsNone(pose('pushups',x))
 def test_untrained_protocol_is_held(self):
  from recognizer import InitialPoseAnalyzer
  a=InitialPoseAnalyzer({'schema':'reviewed_initial_ready_and_return_state_v3','heads':{}})
  r=a.run('ytw',np.zeros((2,33,2)),[0.,.1],view='front',protocol='invented_arms_down_each_rep');self.assertIsNone(r['grade']);self.assertIn('unassessed',r['status'])

class FixedLearnedClass:
 classes_=np.array([0,1])
 def predict_proba(self,x):return np.tile([.3,.7],(len(x),1))
 def predict(self,x):return np.ones(len(x),dtype=int)
class ServicePathTests(unittest.TestCase):
 def test_uncalibrated_soft_probability_does_not_hide_learned_ready_class(self):
  from recognizer import InitialPoseAnalyzer
  a=np.full((33,2),np.nan);a[11]=[100,100];a[13]=[100,150];a[15]=[100,200];a[23]=[200,100]
  head=dict(model=FixedLearnedClass(),state_gate_passed=True)
  bundle=dict(schema='reviewed_initial_ready_and_return_state_v3',heads={('pushups','x','target'):head,('pushups','relative_x','target'):head})
  r=InitialPoseAnalyzer(bundle).run('pushups',np.repeat(a[None],5,axis=0),np.arange(5)*.1,view='side',protocol='arm_return_v1')
  self.assertTrue(r['state']['initial_ready_observed']);self.assertAlmostEqual(r['calibration_time'],.2);self.assertIsNone(r['grade'])
  for f in r['frames']:
   if f['feature_bank']=='relative_x':self.assertGreater(f['time'],r['calibration_time'])
 def test_failed_state_candidate_is_not_silently_used(self):
  from recognizer import InitialPoseAnalyzer
  head=dict(model=FixedLearnedClass(),state_gate_passed=False)
  r=InitialPoseAnalyzer(dict(schema='reviewed_initial_ready_and_return_state_v3',heads={('plank','x','target'):head})).run('plank',np.zeros((1,33,2)),[0.],view='side',protocol='support_hold_v1')
  self.assertEqual(r['status'],'unassessed_state_development_gate_failed');self.assertIsNone(r['grade'])

class HeightAnchorTests(unittest.TestCase):
 def test_standing_height_is_preserved_in_initial_pose_comparison(self):
  a=np.full((33,2),np.nan)
  for j,y in [(11,100),(23,200),(25,300),(27,400)]:a[j]=[100,y]
  b=a.copy();b[np.isfinite(b).all(1),1]+=20
  x,y=pose('squats',a),pose('squats',b)
  self.assertIsNotNone(x);self.assertTrue(np.allclose(x[:-2],y[:-2]));self.assertGreater(abs(y[-2]-x[-2]),.1)
 def test_one_visible_leg_is_enough_for_standing_state_not_both_filled(self):
  a=np.full((33,2),np.nan)
  for j,y in [(11,100),(23,200),(25,300),(27,400)]:a[j]=[100,y]
  self.assertIsNotNone(pose('squats',a));a[27]=np.nan;self.assertIsNone(pose('squats',a))

if __name__=='__main__':unittest.main()
