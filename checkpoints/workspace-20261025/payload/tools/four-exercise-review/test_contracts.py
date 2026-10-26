from pathlib import Path
import unittest,sys,json,hashlib,collections,joblib,numpy as np
sys.dont_write_bytecode=True
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/four-exercise-review-train-v1-20261025';sys.path.insert(0,str(B/'tools/error-enrichment'));from split_guard import assert_training_allowed
class FourContracts(unittest.TestCase):
 @classmethod
 def setUpClass(c):c.rows=json.loads((O/'score-inputs.frozen.json').read_text());c.report=json.loads((O/'score-training.json').read_text());c.art=joblib.load(O/'score-candidates.joblib');c.byid={r['id']:r for r in c.rows}
 def test_frozen_holdout_source_cannot_be_disguised_as_Train(self):
  p=json.loads((B/'data/r2d2-team-20261023/annotations/per-person-stratified-80-20.frozen.json').read_text())
  with self.assertRaises(ValueError):assert_training_allowed([dict(id='renamed_protected_source',split='train',sources=[p['protected_source_sha256'][0]])])
  with self.assertRaises(ValueError):assert_training_allowed([dict(id='fake',split='test',sources=[])])
  assert_training_allowed(self.rows);self.assertEqual({r['exercise'] for r in self.rows},{'pushups','shoulder','ytw','plank'})
 def test_every_candidate_and_baseline_fold_separates_known_sources_and_people(self):
  for h in self.report['heads']:
   for a in h.get('algorithms',[]):
    for f in a['folds']:
     va=[self.byid[i] for i in f['validation_ids']];vs={s for r in va for s in r['sources']};vp={r['person'] for r in va}
     for role in ['audited','old_labels_baseline']:
      tr=[self.byid[i] for i in f[role]['train_ids']];self.assertFalse(vs&{s for r in tr for s in r['sources']});self.assertFalse(vp&{r['person'] for r in tr});self.assertFalse(set(f['held_groups'])&set(f[role]['train_groups']))
 def test_each_model_uses_only_its_observed_bank_and_supported_grades(self):
  for h in self.art['models'].values():
   used=[]
   for i in h['trained_ids']:
    r=self.byid[i];j=next(j for j,v in enumerate(r['items']) if (v['item'],v['bank'])==(h['item'],h['bank']));x=r['items'][j];self.assertTrue(x['eligible']);self.assertIn(r['levels'][j],[0,1,2]);used.append((r,x))
   self.assertEqual({len(x['feature']) for r,x in used},{h['model'].n_features_in_})
   for g in h['grade_values']:self.assertGreaterEqual(h['grade_support'][str(g)]['groups'],2)
   pred=h['model'].predict(np.array([x['feature'] for r,x in used[:3]]));self.assertTrue(set(pred)<=set(h['grade_values']))
 def test_bad_tracking_is_not_taught_as_severity(self):
  for i in ['qevd_00278592_held2_score','qevd_00258746_held2_score']:
   r=self.byid[i];self.assertEqual(r['levels'][3],0);self.assertTrue(all(not x['eligible'] for x in r['items']));self.assertFalse(any(i in h['trained_ids'] for h in self.art['models'].values()))
 def test_incomplete_return_is_neither_error_nor_normal(self):
  for i in ['subject_023_push_up_bad_side','qevd_00215302_existing_review']:
   r=self.byid[i]
   for x,g in zip(r['items'],r['levels']):
    if x['item']=='return_extension':self.assertIsNone(g);self.assertFalse(x['eligible'])
 def test_same_recording_normal_error_are_not_blanket_labels(self):
  r=self.byid['dvids_navy805472_599_689'];q=self.byid['held2_navy805472_2757_2853_held2_score'];self.assertTrue(set(r['sources'])&set(q['sources']));self.assertEqual(r['levels'][0],0);self.assertEqual(q['levels'][0],2);self.assertEqual(r['levels'][2],0);self.assertEqual(q['levels'][2],2)
 def test_no_live_adoption_and_exact_fit_receipt(self):
  self.assertFalse(self.art['live_deployed']);self.assertFalse(self.report['live_deployed']);self.assertEqual(self.art['inputs_sha256'],hashlib.sha256((O/'score-inputs.frozen.json').read_bytes()).hexdigest());self.assertEqual(sum(v['actual_fit_unique_rows'] for v in self.report['summary'].values()),531)
if __name__=='__main__':unittest.main(verbosity=2)
