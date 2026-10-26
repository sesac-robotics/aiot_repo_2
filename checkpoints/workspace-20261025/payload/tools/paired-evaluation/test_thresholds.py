import unittest,json,sys
from pathlib import Path
import numpy as np
from thresholds import fit,predict,transform_input,allowed_indices
B=Path('/home/jayden/SeSACPAI-Mid')
class ThresholdTests(unittest.TestCase):
 def rows(self):return [dict(id=f's{i}',person=f'person{i}',group=f'person{i}',family='synthetic',sources=[],split='train',x=[float(i)],target=1 if i<3 else 2) for i in range(6)]
 def test_untrained_severe_grade_is_not_invented(self):
  r=fit(self.rows(),0,1);self.assertTrue(set(predict(r,[[-100],[100]])).issubset({1.,2.}))
 def test_validation_cannot_calibrate(self):
  rows=self.rows();rows[0]['split']='validation'
  with self.assertRaises(ValueError):fit(rows,0,1)
 def test_protected_source_cannot_calibrate_even_if_renamed(self):
  p=json.loads((B/'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json').read_text());rows=self.rows();rows[0]['sources']=[p['protected_source_sha256'][0]]
  with self.assertRaises(ValueError):fit(rows,0,1)
 def test_native_probability_is_not_a_rule_measurement(self):
  x=np.array([[.1,.2,.3,.99]]);y=x.copy();y[:,-1]=.01
  a=transform_input('side_semantic_roi_v1_aihub_transfer_v1','trunk_line',x);b=transform_input('side_semantic_roi_v1_aihub_transfer_v1','trunk_line',y);self.assertTrue(np.array_equal(a,b));self.assertEqual(a.shape[1],3)
 def test_continuity_depends_on_motion_not_absolute_knee_angle(self):
  a=np.array([[130,135,140,145,150,155,160.]])
  self.assertTrue(np.array_equal(transform_input('side_video','support_continuity',a),transform_input('side_video','support_continuity',a+15)))
 def test_three_support_conditions_are_required_together(self):
  x=np.ones((1,22));x[0,[1,8,15]]=[180,.2,0.]
  self.assertEqual(float(transform_input('side_support_clearance_v1','leg_and_body_support',x)[0,0]),0.)
 def test_persisted_model_integrity_ignores_only_unused_record_padding(self):
  from artifact_integrity import equal_state
  dtype=np.dtype({'names':['value'],'formats':['i4'],'offsets':[0],'itemsize':8})
  a=np.ndarray(2,dtype=dtype,buffer=bytearray([0]*16));b=np.ndarray(2,dtype=dtype,buffer=bytearray([255]*16));a['value']=[1,2];b['value']=[1,2]
  self.assertTrue(equal_state(a,b));b['value'][1]=3;self.assertFalse(equal_state(a,b))
 def test_return_is_terminal_phase_not_earlier_normal_pose(self):
  sys.path.insert(0,str(B/'five-exercise-comparison'))
  import three_items
  seq=np.full((20,33,2),np.nan)
  for n in range(20):
   angle=np.deg2rad(180 if n<15 else 100)
   seq[n,11]=[100,100];seq[n,13]=[100,200];seq[n,15]=[100+100*np.sin(angle),200-100*np.cos(angle)]
  result=three_items.describe('pushups',seq,None)[2]
  self.assertTrue(result['eligible']);self.assertAlmostEqual(result['measurement'],100.,places=5)
 def test_paired_cohort_keeps_identical_targets_and_denominators(self):
  for folder in ['ai-rule-paired-v2-20261024','paired-enriched-five-v2-20261024','paired-failure-enriched-five-v2-20261024']:
   r=json.loads((B/'output'/folder/'comparison.frozen.json').read_text());ps=r['paired_predictions'];self.assertEqual(len(ps),153);self.assertEqual(len({(p['id'],p['criterion']) for p in ps}),153)
   for field in ['same_bank_by_exercise','existing_rules_by_exercise']:
    for pair in r[field].values():
     for key in ['n','normal_n','error_n']:self.assertEqual(pair['ai'][key],pair['rule_base'][key])
if __name__=='__main__':unittest.main()
