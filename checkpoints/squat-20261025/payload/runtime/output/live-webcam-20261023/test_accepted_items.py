import unittest
from pathlib import Path
import numpy as np
from accepted_items import evaluate
class AcceptedTest(unittest.TestCase):
 def test_only_improved_components_and_no_invented_total(self):
  x=np.load(Path(__file__).parent.parent/'squat-relabel-expanded-20261023/observations/Mima_01/front.npz')['image_xy']
  r=evaluate(x)
  self.assertEqual([(i['id'],i['method']) for i in r['items']],[('trunk_balance','ai'),('heel_support','rule')])
  self.assertIsNone(r['total'])
  x=x.copy();x[:,[27,28,29,30,31,32]]=np.nan
  r=evaluate(x)
  self.assertIsNotNone(r['items'][0]['grade'])
  self.assertIsNone(r['items'][1]['grade'])
