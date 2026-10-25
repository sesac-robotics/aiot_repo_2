import json,sys,unittest,importlib.util
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[2];O=B/'output/return-initial-context-20261024';sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'tools/paired-evaluation')]
from thresholds import allowed_indices
spec=importlib.util.spec_from_file_location('upper',B/'tools/return-root-cause/features.py');upper=importlib.util.module_from_spec(spec);spec.loader.exec_module(upper)
read=lambda p:json.loads(p.read_text())
class ContextTests(unittest.TestCase):
 def test_original_targets_and_roles_unchanged(self):
  for file in ['score-inputs.frozen.json','test-inputs.frozen.json']:
   for a,b in zip(read(O/file),read(B/'output/return-discrimination-ensemble-20261024'/file)):
    for k in ['id','person','group','split','sources']:self.assertEqual(a[k],b[k])
    self.assertEqual(a['levels'][:len(b['levels'])],b['levels']);self.assertEqual(a['items'][:len(b['items'])],b['items'])
 def test_normal_r05_uses_observed_full_return_context(self):
  r=next(a for a in read(O/'feature-provenance.json') if a['id']=='return_squat_diag_w2_5_7_r05');self.assertEqual(r['return_source_frame'],246);self.assertGreater(r['return_source_frame']+6,r['original_attempt_end_frame']);self.assertLessEqual(r['return_source_frame']+6,r['context_end_frame'])
 def test_initial_reference_verified_before_attempts(self):
  ls={r['id']:r for r in read(B/'output/squat-return-20261023/labels-v2.frozen.json')['labels']}
  for a in read(O/'feature-provenance.json'):
   if a['id'].startswith(('return_squat_','reviewed_return_squat_')):
    d=ls[a['id'].removeprefix('reviewed_').removeprefix('return_')];self.assertLess(a['reference_window'][1],d['first_frame'])
    if 'diag_m4' not in a['id']:self.assertEqual(a['reference_window'][0],0)
 def test_bounded_context_never_changes_target_bottom_to_next_repetition(self):
  phase=np.array([0,0,.15,.4,.8,.6,.4,.15,0,0,.2,.5,.8,.6,.2,-.2,-.2,-.2]);x=np.zeros((len(phase),33,2));x[:,:,0]=100;x[:,[11,12],1]=(100+100*phase)[:,None];x[:,[23,24],1]=(200+100*phase)[:,None];ref=dict(hip_y=200,shoulder_y=100,center_x=100,scale=100,torso_x=0);a=upper.extract(x,ref);self.assertIsNotNone(a);self.assertLess(a['return_index'],12);self.assertLess(a['bottom_index'],8)
 def test_each_record_context_is_point_four_seconds(self):self.assertTrue(all(a['context_seconds']==.4 for a in read(O/'feature-provenance.json')))
 def test_height_rule_uses_only_height_deficits(self):self.assertEqual(allowed_indices('standing_initial_context_v1','standing_height_return',10),[0,1,2,3])
 def test_six_error_ids_kept(self):
  a=read(O/'new-head-training.json');b=read(B/'output/return-discrimination-ensemble-20261024/new-head-training.json');self.assertEqual({p['id'] for p in a['validation_predictions'] if p['target']<2},{p['id'] for p in b['validation_predictions'] if p['target']<2})
 def test_every_fitted_member_excludes_held_people_and_sources(self):
  rows={r['id']:r for r in read(O/'score-inputs.frozen.json')};h=read(O/'new-head-training.json')
  for fold in h['nested_CV_folds']:
   self.assertFalse(set(fold['fit_groups'])&set(fold['held_groups']))
   for m in fold['inner_members']:
    self.assertFalse(set(m['fit_groups'])&set(m['held_groups']));self.assertTrue(set(m['fit_ids'])<=set(fold['fit_ids']));self.assertFalse(set(m['fit_ids'])&set(fold['outer_ids']))
  for m in h['component_training']:
   self.assertTrue(all(rows[i]['split']=='train' for i in m['fit_ids']));self.assertFalse({s for i in m['fit_ids'] for s in rows[i]['sources']}&{s for i in m['held_ids'] for s in rows[i]['sources']})
if __name__=='__main__':unittest.main()
