import sys,unittest,importlib.util
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;A=O.parents[1]/'output/live-webcam-20261023';sys.path[:0]=[str(O),str(A)]
spec=importlib.util.spec_from_file_location('checklist_comparison',O/'checklist_comparison_candidate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.ROOT=A;sys.modules['checklist_comparison']=m
from server import score_samples
from test_motion_recovery import pose
from standing_state import make_reference
class ItemIsolationTest(unittest.TestCase):
 def test_no_knees_does_not_abort_core_items(self):
  seq=np.array([pose()]*10+[pose(.3)]*8+[pose()]*8);ref=make_reference(seq[:6]);seq[:,[25,26]]=np.nan
  samples=list(enumerate(seq));r=score_samples(samples,640,480,'squats','front',standing_reference=ref,return_samples=samples)
  items={i['id']:i for i in r['score_100']['items']}
  self.assertIsNone(items['descent_range']['ai_points']);self.assertIsNone(items['knee_inward']['ai_points'])
  self.assertIsNotNone(items['standing_return']['ai_points']);self.assertIsNotNone(items['standing_return']['rule_points']);self.assertIsNotNone(items['trunk_balance']['ai_points'])
  self.assertIsNone(r['score_100']['ai']['total']);self.assertEqual(r['score_100']['ai']['assessed_weight'],40)
 def test_missing_core_cannot_be_filled_by_item_isolation(self):
  seq=np.array([pose()]*10+[pose(.3)]*8+[pose()]*8);ref=make_reference(seq[:6]);seq[:,[11,12,23,24]]=np.nan
  samples=list(enumerate(seq));r=score_samples(samples,640,480,'squats','front',standing_reference=ref,return_samples=samples)
  self.assertTrue(all(i['ai_points']is None and i['rule_points']is None for i in r['score_100']['items']))
if __name__=='__main__':unittest.main()
