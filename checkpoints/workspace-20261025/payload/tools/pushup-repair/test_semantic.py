import unittest
import numpy as np
from semantic_features import describe,select_side
class Semantic(unittest.TestCase):
 def body(self):
  a=np.full((20,33,2),np.nan);c=np.ones((20,33,2))
  for j,p in {11:(80,20),13:(80,50),15:(80,80),23:(50,35),25:(25,47.5),27:(0,60),31:(0,80)}.items():a[:,j]=p
  return a,c
 def test_trunk_does_not_depend_on_knee(self):
  a,c=self.body();f=describe(a,c)[0]['feature'];a[:,25]=[20,80];self.assertEqual(f,describe(a,c)[0]['feature'])
 def test_transform_invariance(self):
  a,c=self.body();x=describe(a,c);y=describe(a*2+np.array([100,200]),c)
  for f,g in zip(x,y):self.assertTrue(np.allclose(f['feature'],g['feature']))
 def test_all_missing_is_held(self):
  a,c=self.body();a[:]=np.nan;self.assertTrue(all(not r['eligible'] and r['feature'] is None for r in describe(a,c)))
 def test_side_uses_confidence_not_straightness(self):
  a,c=self.body()
  for j in [11,13,15,23,25,27,31]:a[:,j+1]=a[:,j]
  c[:,::2]=.6;c[:,1::2]=.9;a[:,24]=[50,5]
  self.assertEqual(select_side(a,c,[11,23,27])[0],0)
if __name__=='__main__':unittest.main()
