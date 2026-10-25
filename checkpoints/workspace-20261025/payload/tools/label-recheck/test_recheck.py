import unittest,json,sys
from pathlib import Path
import numpy as np
from features import shape,side_body
B=Path('/home/jayden/SeSACPAI-Mid');sys.path.insert(0,str(B/'tools/error-enrichment'))
from split_guard import assert_training_allowed

class Recheck(unittest.TestCase):
 def make_body(self):
  x=np.full((20,33,2),np.nan)
  for j,p in {11:(0,0),23:(50,0),25:(75,0),27:(100,0)}.items():x[:,j]=p
  return x
 def test_alignment_is_camera_roll_invariant(self):
  x=self.make_body();theta=.43;r=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
  self.assertTrue(np.allclose(side_body(x),side_body(x@r.T+np.array([380,220])),atol=1e-6))
 def test_bent_body_remains_visible_after_normalization(self):
  x=self.make_body();straight=side_body(x);x[:,23,1]=30;bent=side_body(x)
  self.assertGreater(bent[2],straight[2]);self.assertGreater(bent[7],straight[7])
 def test_no_missing_arm_is_invented(self):self.assertIsNone(shape(np.full((20,33,2),np.nan),'W'))
 def test_review_keeps_all_sources_and_no_new_test_targets(self):
  old=B/'output/per-person-stratified-80-20-20261024';new=B/'output/label-verified-five-20261024'
  a=json.loads((old/'test-inputs.frozen.json').read_text());b=json.loads((new/'test-inputs.frozen.json').read_text());a={r['id']:r for r in a};self.assertEqual(set(a),{r['id'] for r in b})
  for r in b:self.assertEqual(r['levels'][:len(a[r['id']]['levels'])],a[r['id']]['levels'])
  labels=json.loads((B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json').read_text())['labels'];self.assertEqual(len({r['source_id'] for r in labels}),57)
 def test_merged_training_window_stays_on_training_side(self):
  r=next(r for r in json.loads((B/'output/label-verified-five-20261024/score-inputs.frozen.json').read_text()) if r['id']=='r2d2_V38_014000_025750')
  assert_training_allowed([r]);self.assertIsNone(r['levels'][0]);self.assertEqual(r['levels'][1:4],[0,2,2])
 def test_training_does_not_admit_any_test_source(self):
  rows=json.loads((B/'output/label-verified-five-20261024/test-inputs.frozen.json').read_text())
  for r in rows:
   with self.assertRaises(ValueError):assert_training_allowed([dict(r,split='train')])
if __name__=='__main__':unittest.main()
