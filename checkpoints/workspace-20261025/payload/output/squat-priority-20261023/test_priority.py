import unittest,json,hashlib,sys
from pathlib import Path
import joblib,numpy as np
O=Path(__file__).resolve().parent;R=O.parents[1]
class AuditTests(unittest.TestCase):
 def test_candidate_has_no_fixed_validation_training_people(self):
  report=json.loads((O/'comparison.json').read_text());bundle=joblib.load(O/'candidate.joblib');rows={r['id']:r for r in report['feature_rows']}
  for key,head in bundle['heads'].items():
   people={rows[i]['person'] for i in head['train_ids']}
   self.assertFalse(people&set(head['validation_people']))
   for i in head['train_ids']:self.assertNotEqual(rows[i]['split'],'validation')
 def test_group_folds_have_no_source_or_person_overlap(self):
  report=json.loads((O/'comparison.json').read_text());rows={r['id']:r for r in report['feature_rows']}
  for head in report['heads']:
   for fold in head['folds']:
    held={s for i in fold['held_ids'] for s in rows[i]['source']};train={s for row in rows.values() if row['person'] in fold['train_people'] for s in row['source']}
    self.assertFalse(held&train);self.assertFalse(set(fold['train_people'])&set(fold['held_people']))
 def test_frozen_label_and_model_provenance(self):
  r=json.loads((O/'comparison.json').read_text());b=joblib.load(O/'candidate.joblib')
  self.assertEqual(hashlib.sha256((O/'labels.frozen.json').read_bytes()).hexdigest(),r['labels_sha256']);self.assertEqual(b['label_sha256'],r['labels_sha256'])
  self.assertEqual(hashlib.sha256((O/'candidate.joblib').read_bytes()).hexdigest(),r['candidate_sha256'])
 def test_fixed_predictions_are_reproducible(self):
  r=json.loads((O/'comparison.json').read_text());b=joblib.load(O/'candidate.joblib');rows={x['id']:x for x in r['feature_rows']}
  for h in r['heads']:
   key=h['item'];m=b['heads'][key]['model']
   for p in h['fixed_validation_predictions']:
    self.assertAlmostEqual(m.predict([rows[p['id']]['new_features'][key]['x']])[0],p['after_ai'],places=10)
 def test_no_auxiliary_label_becomes_full_knee_foot_target(self):
  r=json.loads((O/'labels.frozen.json').read_text())
  for row in r['labels']:
   self.assertIsNone(row['knee_foot']);self.assertIsNone(row['return_extension']);self.assertIsNone(row['original100_levels'][2])
 def test_unreviewed_queue_excludes_only_direct_reviews(self):
  a=json.loads((O/'labels.frozen.json').read_text())['decisions'];b=json.loads((O/'new-people/decisions.json').read_text());q=json.loads((O/'remaining-review.json').read_text());reviewed={r['id'] for r in a+b}
  self.assertEqual(len(reviewed),96);self.assertEqual(len(q['remaining']),4150);self.assertFalse(reviewed&{r['id'] for r in q['remaining']})
if __name__=='__main__':unittest.main()
