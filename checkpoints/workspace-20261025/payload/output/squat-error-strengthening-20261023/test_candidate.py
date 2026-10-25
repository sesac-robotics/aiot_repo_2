import unittest,json,hashlib
from pathlib import Path
import numpy as np,joblib
from predict import predict
from features import stance_context
O=Path(__file__).resolve().parent
class CandidateTests(unittest.TestCase):
 def test_person_and_source_separation(self):
  d=json.loads((O/'comparison.json').read_text());rows=json.loads((O/'feature-rows.json').read_text())
  for f in d['folds']:
   a=[r for r in rows if r['person'] in f['train_people']];b=[r for r in rows if r['person'] in f['held_people']]
   self.assertFalse({r['person'] for r in a}&{r['person'] for r in b});self.assertFalse({s for r in a for s in r['source']}&{s for r in b for s in r['source']})
 def test_fixed_validation_person_not_fitted(self):
  d=joblib.load(O/'candidate.joblib')
  for key,group in d['heads'].items():
   for h in group.values():
    if h:self.assertNotIn('rehab26_Mima',h['training_people'])
 def test_labels_frozen_and_not_prediction_targets(self):
  raw=(O/'labels.frozen.json').read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),(O/'labels.sha256').read_text());rs=json.loads(raw)['records'];self.assertEqual(len(rs),24)
  self.assertTrue(all(not r['model_outputs_used'] and not r['rule_generated'] for r in rs));self.assertTrue(all(r['levels']['standing_return']==2 for r in rs))
 def test_head_ankle_absence_keeps_core_predictions(self):
  z=np.load(O/'observations/Petra_a10_0/front.npz');x=z['image_xy'].copy();a=predict(x,int(z['width']),int(z['height']));x[:,:11]=np.nan;x[:,27:]=np.nan;b=predict(x,int(z['width']),int(z['height']))
  self.assertIsNone(stance_context(x));self.assertEqual(b['items'][0]['route'],'core_without_ankles');self.assertIsNotNone(b['items'][0]['ai_grade']);self.assertAlmostEqual(a['items'][1]['ai_grade'],b['items'][1]['ai_grade'],places=10);self.assertIsNone(b['total'])
 def test_no_normal_label_for_missing_required_knees(self):
  z=np.load(O/'observations/Petra_a10_0/front.npz');x=z['image_xy'].copy();x[:,25:27]=np.nan;r=predict(x,int(z['width']),int(z['height']));self.assertIsNone(r['items'][0]['ai_grade']);self.assertIsNotNone(r['items'][1]['ai_grade'])
 def test_existing_two_misses_reported_not_hidden(self):
  d=json.loads((O/'comparison.json').read_text());ps=[r for r in d['items']['knee_inward']['predictions'] if r['id'] in ['Vasil_02','Vasil_08']];self.assertEqual(len(ps),2);self.assertTrue(all(p['context_ai']>=1.5 for p in ps));self.assertFalse(d['original_test_person_opened'])
if __name__=='__main__':unittest.main()
