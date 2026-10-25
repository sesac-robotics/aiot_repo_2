from pathlib import Path
import json,unittest,hashlib
import joblib
O=Path(__file__).resolve().parent;P=O.parent/'squat-priority-20261023'
def read(p):return json.loads(p.read_text())
class Audit(unittest.TestCase):
 def test_metrics_count_real_held_errors(self):
  for file,field in [('error-focus-comparison.json','pred'),('rule-focus-comparison.json','prediction')]:
   for h in read(O/file)['reports']:
    self.assertFalse(h['improved'])
    for phase in ['before','after']:
     ps=h[phase+'_predictions'];m=h[phase];self.assertEqual(m['n'],len(ps));self.assertEqual(m['missed_errors'],sum(p['target']<2 and p[field]>=1.5 for p in ps));self.assertEqual(m['normal_flagged'],sum(p['target']==2 and p[field]<1.5 for p in ps))
 def test_person_and_source_disjoint_folds(self):
  rows=read(P/'comparison.json')['feature_rows']
  for file in ['error-focus-comparison.json','rule-focus-comparison.json']:
   for h in read(O/file)['reports']:
    for f in h['folds']:
     train=set(f['train_people']);held=set(f['held_people']);self.assertFalse(train&held);a={s for r in rows if r['person'] in train for s in r['source']};b={s for r in rows if r['person'] in held for s in r['source']};self.assertFalse(a&b)
 def test_final_fit_excludes_historical_validation_people(self):
  rows={r['id']:r for r in read(P/'comparison.json')['feature_rows']};bundle=joblib.load(O/'error-focus-candidate.joblib')
  for head in bundle['heads'].values():
   self.assertFalse(head['runtime_approved'])
   for id in head['train_ids']:self.assertEqual(rows[id]['split'],'train')
 def test_no_new_normals_or_pseudolabels_were_added(self):
  r=read(O/'review-decisions.json');self.assertEqual(r['new_training_labels'],0)
  for row in r['records']:
   self.assertFalse(row['used_for_training']);self.assertIsNone(row['knee_inward']);self.assertIsNone(row['return_extension']);self.assertIsNone(row['trunk_balance'])
 def test_original_models_and_previous_candidate_preserved(self):
  v=read(O/'verification.json')
  for p,sha in v['original_model_sha256'].items():self.assertEqual(hashlib.sha256(Path(p).read_bytes()).hexdigest(),sha)
  self.assertEqual(hashlib.sha256((P/'candidate.joblib').read_bytes()).hexdigest(),v['previous_candidate_sha256'])
 def test_review_queue_has_no_false_completion(self):
  r=read(O/'remaining-review.json');decisions=read(O/'review-decisions.json')['records'];seen={x['id'] for x in decisions};self.assertEqual(len(seen),176);self.assertEqual(len(r['remaining']),3974);self.assertEqual(set(r['pending_adjudication_ids']),seen);self.assertFalse(r['complete_relabelling']);self.assertFalse(seen&{x['id'] for x in r['remaining']})
if __name__=='__main__':unittest.main()
