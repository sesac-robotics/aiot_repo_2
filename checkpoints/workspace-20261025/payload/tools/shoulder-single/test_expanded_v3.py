from pathlib import Path
import unittest,sys,json,hashlib,collections
import numpy as np,joblib
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-expanded-train-v3-20261025';sys.path.insert(0,str(B/'tools/shoulder-work'));sys.path.insert(0,str(B/'tools/error-enrichment'));from observation_features_v4 import extract_body;from split_guard import assert_training_allowed
class ExpandedTrainContracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.rows=json.loads((O/'training-inputs.frozen.json').read_text());cls.report=json.loads((O/'model-validation.json').read_text());cls.artifact=joblib.load(O/'research-models.joblib');cls.sources=json.loads((O/'source-plan.frozen.json').read_text());cls.candidates=json.loads((O/'feature-candidates.json').read_text())
 def test_original_sources_and_Train_roles(self):
  policy=json.loads((B/'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json').read_text());assert_training_allowed(self.rows)
  for r in self.sources:
   self.assertEqual(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest(),r['source_sha256']);self.assertEqual(r['split'],'train')
   if r['family']=='team':self.assertEqual(policy['assignments'][r['id']],'train')
 def test_relabelled_Test_cannot_train(self):
  p=json.loads((B/'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json').read_text());self.assertTrue(p['protected_source_sha256'])
  with self.assertRaises(ValueError):assert_training_allowed([dict(id='tampered',split='train',sources=[p['protected_source_sha256'][0]])])
 def test_exact_cache_identity_and_context(self):
  for r in self.rows:
   p=Path(r['cache_path']);z=np.load(p);m=json.loads(p.with_suffix('.json').read_text());self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),r['cache_sha256']);self.assertEqual(m['config']['source_sha256'],r['source_sha256']);self.assertEqual(m['fingerprint'],r['cache_fingerprint']);self.assertTrue(set(r['all_context_native_frames'])<=set(z['frame_numbers']));self.assertTrue(r['visual_qa']['all_actual_context_frames_reviewed'])
 def test_body_is_independent_of_arm_or_head_connections(self):
  r=self.candidates[-2];z=np.load(r['cache_path']);ticks=np.rint(z['frame_numbers']/r['fps']*10).astype(int);x=z['image_xy'].copy();x[:,[0,1,2,3,4,5,6,7,8,9,10,13,14,15,16,25,26,27,28,29,30,31,32]]=np.nan;f=extract_body(list(zip(ticks,x)),[round(t*10) for t in r['window_seconds']],True,r['width'],r['height'],'front');self.assertEqual(f['status'],'features_ready');np.testing.assert_allclose(f['features'],r['body']['features'])
 def test_missing_hip_not_a_movement_error(self):
  r=self.candidates[-2];z=np.load(r['cache_path']);ticks=np.rint(z['frame_numbers']/r['fps']*10).astype(int);x=z['image_xy'].copy();x[:,23]=np.nan;f=extract_body(list(zip(ticks,x)),[round(t*10) for t in r['window_seconds']],True,r['width'],r['height'],'front');self.assertEqual(f['status'],'held');self.assertNotIn('grade',f)
 def test_error_person_excluded_means_abstention(self):
  folds=self.report['reports']['body_logistic']['leave_person_out'];err=[r for r in folds if r['truth']==1];self.assertEqual(len(err),2);self.assertTrue(all(r['prediction'] is None for r in err))
  for r in folds:self.assertNotIn(r['person'],r['training_people'])
 def test_feature_schemas_cannot_be_silently_mixed(self):
  ids={r['id']:r for r in self.rows}
  for h in self.artifact['heads'].values():
   used=[ids[i] for i in h['training_ids']];self.assertEqual({r['feature_schema'] for r in used},{h['feature_schema']});self.assertEqual({r['posture'] for r in used},{'seated'});self.assertEqual({r['view'] for r in used},{'front'})
 def test_no_grade0_points_or_production_enablement(self):
  self.assertFalse(self.artifact['grade_prediction_enabled']);self.assertFalse(self.artifact['score_prediction_enabled']);self.assertFalse(self.artifact['production_eligible']);self.assertNotIn(0,{r['grade'] for r in self.rows});self.assertEqual({r['grade'] for r in self.rows if r['item']=='lower_return'},{2})
 def test_source_scope_exclusions_are_kept_out_of_fit(self):
  lab=json.loads((O/'visual-labels.frozen.json').read_text());self.assertEqual(len(lab['team_clips']),31);self.assertTrue(all(r['lower_grade'] is None for r in lab['team_clips'] if r['source_id']=='V25'));self.assertFalse(any(r['source_id'].startswith('youtube') for r in self.rows))
if __name__=='__main__':unittest.main()
