from pathlib import Path
import unittest,json,sys,joblib,numpy as np,hashlib
sys.dont_write_bytecode=True
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/four-exercise-source-phase-v2-20261025/clock-corrected-v3'
class TrainingContracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.rows={r['id']:r for r in json.loads((O/'score-inputs.frozen.json').read_text())};cls.report=json.loads((O/'score-training.json').read_text());cls.bundle=joblib.load(O/'score-candidates.joblib');cls.folds=json.loads((O/'folds.frozen.json').read_text())
 def test_exact_native_schedule_preserves_distinct_15fps_observations(self):
  receipts=json.loads((O/'body-clock-correction-receipts.json').read_text());restored=[r for r in receipts if r['corrected_eligible'] and not r['previous_eligible']];self.assertEqual(len(restored),21)
  for r in receipts:
   v=r['result'];ticks=np.array(v['actual_observation_slots']);fs=np.array(v['actual_native_frames']);self.assertTrue(np.all(np.diff(ticks)>0));self.assertEqual(len(ticks),len(fs));self.assertEqual(v['observation_slot_clock'],'exact_index_in_verified_native_extraction_schedule')
  self.assertEqual(np.rint(np.array([98,100,101])/15*10).astype(int).tolist(),[65,67,67])
 def test_correction_does_not_claim_new_fits_or_change_trained_vectors(self):
  p=O.parent;old=joblib.load(p/'score-candidates.joblib');rows={r['id']:r for r in json.loads((p/'score-inputs.frozen.json').read_text())};self.assertEqual(set(old['models']),set(self.bundle['models']));self.assertEqual(self.report['body_clock_correction']['new_classifier_fits_this_correction'],0)
  for key,h in self.bundle['models'].items():
   ex,item,bank,alg=key.split('|');self.assertEqual(h['trained_ids'],old['models'][key]['trained_ids'])
   for id in h['trained_ids']:
    r=rows[id];j=next(j for j,x in enumerate(r['items']) if (x['item'],x['bank'])==(item,bank));self.assertEqual(r['items'][j]['feature'],self.rows[id]['items'][j]['feature'])
 def test_connected_folds_have_no_person_or_original_source_overlap(self):
  for fold in range(4):
   a=[r for id,r in self.rows.items() if self.folds['fold_by_id'][id]!=fold];b=[r for id,r in self.rows.items() if self.folds['fold_by_id'][id]==fold]
   self.assertFalse({r['person'] for r in a}&{r['person'] for r in b});self.assertFalse({s for r in a for s in r['sources']}&{s for r in b for s in r['sources']})
  for h in self.report['heads']:
   for a in h['algorithms']:
    for f in a['folds']:
     for role in ['candidate','baseline']:
      self.assertTrue(all(self.folds['fold_by_id'][i]!=f['fold'] for i in f[role]['train_ids']))
 def test_each_model_uses_observed_features_and_two_groups_per_grade(self):
  for key,head in self.bundle['models'].items():
   ex,item,bank,alg=key.split('|');vectors=[]
   for id in head['trained_ids']:
    r=self.rows[id];j=next(j for j,x in enumerate(r['items']) if (x['item'],x['bank'])==(item,bank));x=r['items'][j];self.assertEqual(r['split'],'train');self.assertTrue(x['eligible']);self.assertIn(r['levels'][j],head['supported_grades']);vectors.append(x['feature'])
   self.assertEqual({len(x) for x in vectors},{head['model'].n_features_in_});self.assertTrue(np.isfinite(vectors).all())
   for g in head['supported_grades']:self.assertGreaterEqual(head['grade_support'][str(g)]['groups'],2)
   self.assertTrue(set(head['model'].predict(vectors[:3]))<=set(head['supported_grades']))
 def test_baseline_feature_units_are_not_mixed(self):
  rows=json.loads((O/'paired-baseline-used-v2.frozen.json').read_text())
  for bank,size in [('observed_post_descent_return_v2',8),('hinged_stage_contiguous_v4',17),('upright_stage_contiguous_v4',17)]:
   fs=[x['feature'] for r in rows for x in r['items'] if x['bank']==bank and x['eligible'] and x.get('feature') is not None];self.assertEqual({len(x) for x in fs},{size})
 def test_metrics_count_observation_holds_as_abstentions(self):
  for h in self.report['heads']:
   for a in h['algorithms']:
    for role,mkey in [('candidate','candidate_cv'),('baseline','baseline_cv_same_targets')]:
     p=a['predictions'][role];m=a[mkey];self.assertEqual(len(p),h['labelled_targets']);self.assertEqual(m['evaluated']+m['abstained'],m['labelled_targets']);self.assertEqual(m['evaluated'],sum(x['prediction'] is not None for x in p));self.assertEqual(m['errors_detected'],sum(x['target']<2 and x['prediction'] is not None and x['prediction']<2 for x in p))
     self.assertEqual([(r['id'],r['fold'],r['target']) for r in a['predictions']['candidate']],[(r['id'],r['fold'],r['target']) for r in a['predictions']['baseline']])
 def test_sparse_shoulder_errors_are_not_a_supported_classifier(self):
  self.assertFalse(any('four_joint_body_v2/' in k or 'anatomical_lower_turn_v1/' in k for k in self.bundle['models']))
  for h in self.report['heads']:
   if h['exercise']=='shoulder' and h['item'] in ['lower_return','frontal_trunk_hold']:
    self.assertFalse(any(a['development_gate_passed'] for a in h['algorithms']))
 def test_training_claim_matches_serialized_model_ids(self):
  for ex,s in self.report['summary'].items():
   ids={i for key,h in self.bundle['models'].items() if key.startswith(ex+'|') for i in h['trained_ids']};self.assertEqual(ids,set(s['actual_fit_ids']));self.assertEqual(len(ids),s['actual_fit_unique_records'])
  self.assertEqual(self.bundle['inputs_sha256'],hashlib.sha256((O/'score-inputs.frozen.json').read_bytes()).hexdigest());self.assertFalse(self.bundle['live_deployed']);self.assertFalse(self.report['live_deployed'])
if __name__=='__main__':unittest.main(verbosity=2)
