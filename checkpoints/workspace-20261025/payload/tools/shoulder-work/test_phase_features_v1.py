import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'output/live-webcam-20261023'))
from test_shoulder_candidate import pose
from phase_features_v1 import extract
class Tests(unittest.TestCase):
 def seq(self):return [pose()]*6+[pose(True)]*4+[pose()]*3
 def runseq(self,s,ticks=None):return extract(list(zip(range(len(s)) if ticks is None else ticks,s)))
 def test_reference_return(self):
  d=self.runseq(self.seq());self.assertEqual(d['status'],'features_ready');self.assertEqual(d['return_ticks'],[10,11,12]);np.testing.assert_allclose(d['return_features'],0);self.assertIsNone(d['grade'])
 def test_truncated_ascent_no_grade(self):self.assertEqual(self.runseq(self.seq()[:-3])['status'],'held')
 def test_missing_not_return_failure(self):
  s=self.seq();s[8]=np.full((33,2),np.nan);self.assertEqual(self.runseq(s)['status'],'held')
 def test_sparse_time_not_filled(self):self.assertEqual(self.runseq(self.seq(),np.arange(13)*2)['status'],'held')
 def test_border_excluded(self):
  s=[p.copy() for p in self.seq()]
  for p in s:p[23,1]=500
  self.assertEqual(self.runseq(s)['status'],'held')
 def test_duplicate(self):
  with self.assertRaises(ValueError):self.runseq(self.seq(),[0]*13)
 def test_no_head_ankle_needed(self):self.assertEqual(self.runseq(self.seq())['status'],'features_ready')
 def test_actual_next_ascent_partial(self):
  p=pose(True);p[13]=[240,215];p[15]=[235,165];p[14]=[400,215];p[16]=[405,165]
  d=self.runseq([pose()]*6+[pose(True)]*4+[p]*3+[pose(True)]*3)
  self.assertEqual(d['status'],'features_ready');self.assertEqual(d['closure'],'lower_turnaround_with_observed_next_ascent');self.assertIsNone(d['grade'])
 def test_partial_cut_no_failure(self):
  p=pose(True);p[13]=[240,215];p[15]=[235,165];p[14]=[400,215];p[16]=[405,165]
  self.assertEqual(self.runseq([pose()]*6+[pose(True)]*4+[p]*3)['status'],'held')
if __name__=='__main__':unittest.main()
