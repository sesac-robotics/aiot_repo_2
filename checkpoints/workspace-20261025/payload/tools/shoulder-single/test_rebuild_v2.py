from pathlib import Path
import unittest,json,sys,hashlib
import numpy as np,joblib
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-rebuild-v2-20261025';sys.path.insert(0,str(B/'tools/shoulder-work'));from observation_features_v4 import extract_body
class RebuildContracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.rows=json.loads((O/'training-inputs.frozen.json').read_text());cls.candidates={r['id']:r for r in json.loads((O/'event-feature-candidates.json').read_text())};cls.inv={r['id']:r for r in json.loads((B/'output/shoulder-work-20261024/full-review-v1/inventory.frozen.json').read_text())['sources']};cls.artifact=joblib.load(O/'normal-reference-rebuilt.joblib');cls.metrics=json.loads((O/'model-validation.json').read_text())
 def test_original_source_roles_and_hashes_preserved(self):
  for r in self.rows:
   source=self.inv[r['source_id']];self.assertEqual(source['split'],r['split']);self.assertEqual(r['split'],'train');self.assertEqual(r['source_sha256'],source['source_sha256']);self.assertEqual(hashlib.sha256(Path(source['path']).read_bytes()).hexdigest(),source['source_sha256'])
 def test_native_turn_inside_exact_visual_label(self):
  for r in self.rows:
   if r['item']!='lower_return':continue
   a,b=r['original_reviewed_turn_seconds'];fps=self.candidates[r['id'].rsplit('_lower_return',1)[0]]['fps'];start,end=r['event_native_windows']['turn_frames'];self.assertGreaterEqual(start/fps,a-.011);self.assertLessEqual(end/fps,b+.011)
 def test_context_has_only_actual_native_samples(self):
  for r in self.rows:
   z=np.load(r['cache_path']);self.assertTrue(set(r['all_context_native_frames'])<=set(z['frame_numbers']));self.assertTrue(np.all(np.diff(r['all_context_native_frames'])>0));self.assertFalse(r['initial_ready_inferred']);self.assertFalse(r['waiting_or_preparation_included'])
 def test_cache_fingerprint_matches_frozen_input(self):
  for r in self.rows:
   p=Path(r['cache_path']);m=json.loads(p.with_suffix('.json').read_text());self.assertEqual(r['cache_sha256'],hashlib.sha256(p.read_bytes()).hexdigest());self.assertEqual(r['cache_fingerprint'],m['fingerprint']);self.assertEqual(m['config']['source_sha256'],r['source_sha256'])
 def test_manual_qa_rejections_are_not_fit(self):
  rejected=['subject_018_shoulder_bad_front_event_1','subject_013_shoulder_good_front_event_1','subject_006_shoulder_good_front_event_1','subject_007_shoulder_good_front_event_1','subject_026_shoulder_bad_front_event_2'];self.assertFalse(any(any(r['id'].startswith(v) for v in rejected) for r in self.rows))
 def test_body_remains_independent_of_arm_tracking(self):
  r=next(r for r in self.rows if r['id']=='subject_006_shoulder_bad_front_event_2_frontal_trunk_hold');c=self.candidates['subject_006_shoulder_bad_front_event_2'];z=np.load(r['cache_path']);native=z['frame_numbers'];times=native/c['fps'];ticks=np.rint((times-times[0])*10).astype(int)+round(times[0]*10);x=z['image_xy'].copy();x[:,[0,1,2,3,4,5,6,7,8,9,10,13,14,15,16,25,26,27,28]]=np.nan;window=[round(v*10) for v in r['feature_window_seconds']];f=extract_body(list(zip(ticks,x)),window,True,c['width'],c['height'],'front');self.assertEqual(f['status'],'features_ready');np.testing.assert_allclose(f['features'],r['features'])
 def test_people_and_source_groups_excluded_in_folds(self):
  byid={r['id']:r for r in self.rows}
  for head in self.metrics['reports'].values():
   for algo in head.get('comparison',{}).values():
    for fold in algo['leave_person_out']:self.assertNotIn(byid[fold['id']]['person'],fold['training_people'])
 def test_no_missing_grades_turned_into_points(self):
  self.assertEqual({r['grade'] for r in self.rows},{2});self.assertFalse(self.artifact['grade_prediction_enabled']);self.assertFalse(self.artifact['score_prediction_enabled']);self.assertFalse(self.artifact['production_eligible']);self.assertTrue(all(x['points'] is None and not x['grade_prediction_enabled'] for x in self.artifact['capabilities']))
 def test_seated_people_not_pooled_into_standing(self):
  for key,h in self.artifact['heads'].items():
   self.assertEqual(key[1],'standing');ids=set(h['training_ids']);self.assertTrue(all(r['posture']=='standing' for r in self.rows if r['id']in ids))
 def test_new_schema_does_not_reuse_ready_relative_head(self):
  for r in self.rows:
   if r['item']=='lower_return':self.assertEqual(r['criterion'],'observed_anatomical_lower_turn');self.assertEqual(r['feature_schema'],'shoulder-anatomical-lower-turn-v1')
if __name__=='__main__':unittest.main()
