import sys,unittest,copy
import numpy as np
sys.path.insert(0,'/home/jayden/SeSACPAI-Mid/output/live-webcam-20261023')
from test_ytw_candidate import points
from ytw_readiness_v5 import YTWTracker,scoring_input_hold
class ReadinessTests(unittest.TestCase):
 def fixture(self):
  now={'phase':'Y','ready':True};t=YTWTracker(lambda p:(now['phase'],now['ready']));return t,now
 def complete(self):
  t,n=self.fixture();events=[]
  for i,label in enumerate('YYYYYTTTTTWWWWW'):
   n['phase']=label;e=t.push(i,points(label))
   if e:events.append(e)
  return t,events[0]
 def test_pending_then_confirmed_receipts(self):
  t,n=self.fixture();t.push(0,points('Y'));self.assertEqual(t.status()['current_state'],'Y_confirmation_pending')
  for i in range(1,5):t.push(i,points('Y'))
  s=t.status();self.assertTrue(s['current_Y_ready']);self.assertTrue(s['current_Y_boundary_confirmed'])
  self.assertEqual(s['attempt_receipts'],dict(Y_phase_candidate_at_tick=0,Y_ready_confirmed_at_tick=4,Y_stable_confirmed_at_tick=2))
 def test_missing_hides_readiness_preserves_history(self):
  t,n=self.fixture()
  for i in range(5):t.push(i,points('Y'))
  t.push(5,np.full((33,2),np.nan));s=t.status()
  self.assertEqual(s['current_state'],'observation_wait');self.assertFalse(s['current_Y_ready']);self.assertTrue(s['attempt_Y_ready_observed']);self.assertIsNone(s['quality_grade'])
 def test_ready_class_release_requires_fresh_five(self):
  t,n=self.fixture()
  for i in range(5):t.push(i,points('Y'))
  n['ready']=False;t.push(5,points('Y'));self.assertFalse(t.status()['current_Y_ready'])
  n['ready']=True
  for i in range(6,10):t.push(i,points('Y'));self.assertFalse(t.status()['current_Y_ready'])
  t.push(10,points('Y'));self.assertTrue(t.status()['current_Y_ready'])
 def test_gap_does_not_use_old_ready_streak(self):
  t,n=self.fixture()
  for i in range(5):t.push(i,points('Y'))
  t.push(6,points('Y'));self.assertFalse(t.status()['current_Y_ready']);self.assertEqual(t.status()['current_Y_ready_streak'],1)
 def test_T_is_not_current_Y_ready(self):
  t,n=self.fixture()
  for i in range(5):t.push(i,points('Y'))
  n['phase']='T';t.push(5,points('T'));s=t.status()
  self.assertFalse(s['current_Y_ready']);self.assertTrue(s['attempt_Y_ready_observed']);self.assertEqual(s['current_state'],'ordered_attempt_in_progress')
 def test_unknown_model_holds(self):
  t=YTWTracker(lambda p:(None,False));t.push(0,points('Y'));self.assertEqual(t.status()['current_state'],'state_model_unavailable');self.assertEqual(t.count,0)
 def test_finish_is_not_failure_grade(self):
  t,n=self.fixture()
  for i in range(5):t.push(i,points('Y'))
  s=t.finish();self.assertEqual(s['current_state'],'observation_ended');self.assertFalse(s['current_Y_ready']);self.assertIsNone(s['quality_grade']);self.assertEqual(t.count,0)
 def test_complete_has_distinct_timestamps(self):
  t,e=self.complete();r=e['readiness_receipts']
  self.assertEqual((r['Y_phase_candidate_at_tick'],r['Y_ready_confirmed_at_tick'],r['Y_stable_confirmed_at_tick'],r['ordered_sequence_confirmed_at_tick']),(0,4,2,14))
  self.assertFalse(t.status()['current_Y_ready']);self.assertEqual(t.count,1)
 def test_T_W_phase_confirmation_is_not_physical_entry(self):
  t,e=self.complete();r=e["readiness_receipts"]
  self.assertEqual((r["T_phase_candidate_at_tick"],r["T_phase_confirmed_at_tick"],r["W_phase_candidate_at_tick"],r["W_phase_confirmed_at_tick"]),(5,9,10,14))
 def test_unstable_Y_boundary_does_not_grant_input_contract(self):
  n={"label":"Y"};t=YTWTracker(lambda p:(n["label"],True));events=[]
  for i,label in enumerate("Y"*10+"T"*5+"W"*5):
   n["label"]=label;p=points(label)
   if i<10:p[[15,16],0]+=40 if i%2 else -40
   e=t.push(i,p)
   if e:events.append(e)
  self.assertEqual(t.count,1);self.assertFalse(scoring_input_hold(events[0])["input_contract_eligible"])
 def test_contract_keeps_manual_permissions_false(self):
  t,e=self.complete();d=scoring_input_hold(e);self.assertTrue(d['input_contract_eligible']);self.assertFalse(d['full_score_authorized']);self.assertFalse(d['existing_engine_options']['complete_attempt'])
 def test_no_boundary_is_held(self):
  t,e=self.complete();e['auto_phase_windows']=None;e['Y_boundary_evidence']['confirmed']=False
  d=scoring_input_hold(e);self.assertFalse(d['input_contract_eligible']);self.assertIn('Y_boundary_unconfirmed',d['holds'])
 def test_invalid_window_dense_border_or_permission_is_held(self):
  t,e=self.complete()
  for mutation in ['window','dense','border','permission','protocol']:
   x=copy.deepcopy(e)
   if mutation=='window':x['auto_phase_windows']['Y']=[-1,0]
   if mutation=='dense':x['samples'].pop(1)
   if mutation=='border':x['samples'][0][1][24]=[639,350]
   if mutation=='permission':x['complete_attempt_reviewed']=True
   if mutation=='protocol':x['candidate_context']['protocol']='hinged'
   self.assertFalse(scoring_input_hold(x)['input_contract_eligible'],mutation)
 def test_head_ankle_not_required(self):
  t,e=self.complete();self.assertEqual(t.count,1);self.assertEqual(e['observation_coverage'],1)
 def test_malformed_timestamp_leaves_receipt(self):
  t,n=self.fixture();t.push(0,points('Y'));old=copy.deepcopy(t.status())
  with self.assertRaises(ValueError):t.push(0,points('Y'))
  self.assertEqual(t.status(),old)
if __name__=='__main__':unittest.main()
