import sys,unittest,pickle,json
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[2];sys.path.insert(0,str(B/'tools/error-enrichment'))
from models import ScaledRegressor,NormalThresholdRegressor
class CalibrationTests(unittest.TestCase):
 def setUp(self):
  self.base=ScaledRegressor('grade1').fit([[0],[.1],[1],[1.1]],[2,2,1,1]);self.base.project_grade_min=1;self.base.project_grade_max=2
 def test_observed_grades_only(self):
  p=NormalThresholdRegressor(self.base,.5).predict([[0],[1.1]]);self.assertEqual(set(p),{1,2})
 def test_cutoff_changes_border(self):
  self.assertEqual(NormalThresholdRegressor(self.base,0).predict([[.5]])[0],2);self.assertEqual(NormalThresholdRegressor(self.base,1).predict([[.5]])[0],1)
 def test_serialization(self):
  m=NormalThresholdRegressor(self.base,.7);self.assertTrue(np.array_equal(m.predict([[.2],[.7]]),pickle.loads(pickle.dumps(m)).predict([[.2],[.7]])))
 def test_nested_groups_and_ids_disjoint(self):
  heads=json.loads((B/'output/mild-return-calibrated-20261024/changed-heads-fit/score-training.json').read_text())['heads'];rows=json.loads((B/'output/mild-return-calibrated-20261024/score-inputs.frozen.json').read_text());tr={r['id']:r for r in rows if r['split']=='train'}
  for h in heads:
   for f in h['nested_CV_folds']:
    self.assertFalse(set(f['cutoff_fit_ids'])&set(f['outer_ids']));self.assertFalse(set(f['held_groups'])&set(f['fit_groups']));self.assertTrue(set(f['cutoff_fit_ids'])<=tr.keys());a={s for i in f['cutoff_fit_ids'] for s in tr[i]['sources']};b={s for i in f['outer_ids'] for s in tr[i]['sources']};self.assertFalse(a&b)
if __name__=='__main__':unittest.main()
