import unittest,numpy as np
from cycle_features import locate
class CycleTiming(unittest.TestCase):
 def test_first_partial_return_cannot_be_replaced_by_later_full_return(self):
  a=[175,175,160,130,100,100,120,140,150,150,140,110,100,120,150,175,178,178,178]
  r=locate(a,np.ones(len(a),bool));self.assertIsNotNone(r);self.assertLess(r['return_index'],12)
 def test_next_descent_tail_not_mistaken_for_return(self):
  a=[178,178,160,130,100,100,120,155,178,179,170,140,110,95]
  r=locate(a,np.ones(len(a),bool));self.assertIsNotNone(r);self.assertGreater(a[r['return_index']],170)
 def test_cut_while_rising_is_held(self):
  a=[175,175,160,130,100,95,110,125,140]
  self.assertIsNone(locate(a,np.ones(len(a),bool)))
 def test_bottom_in_appended_context_is_not_original_attempt(self):
  a=[178,178,160,130,100,100,120,155,178,178,178]
  core=np.arange(len(a))<4;self.assertIsNone(locate(a,core))
 def test_standing_still_is_not_a_repetition(self):self.assertIsNone(locate([178]*12,np.ones(12,bool)))
if __name__=='__main__':unittest.main()
