import unittest
from pathlib import Path
import numpy as np
from method_comparison import compare_candidate
from front_candidate import predict
class ChecklistComparisonTests(unittest.TestCase):
 def segment(self):
  z=np.load(Path(__file__).parent/'recordings/20261023-110200-964840/pose.npz');return z['image_xy'][35:86].copy()
 def test_same_labels_exact_model_reconstruction_and_rule(self):
  x=self.segment();r=compare_candidate(predict(x),x)
  for i in r['item_comparisons']:
   if i['item'] not in ('knee_inward','trunk_balance'):continue
   self.assertIsNotNone(i['ai_grade']);m=i['model_evidence'];self.assertAlmostEqual(m['baseline']+sum(v['contribution'] for v in m['contributions']),m['prediction'],places=6)
   e=i['rule_evidence'];v=e['value'];good,partial=e['boundaries'];higher=e['higher'];expected=2 if (v>=good if higher else v<=good) else 1 if (v>=partial if higher else v<=partial) else 0
   self.assertEqual(i['rule_grade'],expected);self.assertEqual(i['criterion_version'],'squat-observation-checklist-v2')
 def test_missing_foot_does_not_change_other_items(self):
  x=self.segment();a=compare_candidate(predict(x),x);x[:,27:]=np.nan;b=compare_candidate(predict(x),x)
  self.assertEqual([(i.get('ai_grade'),i.get('rule_grade')) for i in a['item_comparisons']],[(i.get('ai_grade'),i.get('rule_grade')) for i in b['item_comparisons']])
 def test_missing_knee_not_fabricated_and_body_continues(self):
  x=self.segment();c=predict(x);x[:,25:27]=np.nan;r=compare_candidate(c,x);items={i['item']:i for i in r['item_comparisons']}
  self.assertIsNone(items['knee_inward']['ai_grade']);self.assertIsNotNone(items['trunk_balance']['ai_grade'])
 def test_return_never_claims_error_trained_ai(self):
  x=self.segment();r=compare_candidate(predict(x),x);ret=next(i for i in r['item_comparisons'] if i['item']=='standing_return')
  self.assertIsNone(ret['ai_grade']);self.assertIsNotNone(ret['rule_grade']);self.assertEqual(ret['status'],'insufficient_error_labels')
 def test_saved_evidence_reproduces_heads_without_relabeling(self):
  x=self.segment();c=predict(x);a=compare_candidate(c,x);b=compare_candidate(c,evidence=a['feature_evidence'])
  self.assertEqual([(i['item'],i['status'],i.get('ai_grade'),i.get('rule_grade')) for i in a['item_comparisons']],[(i['item'],i['status'],i.get('ai_grade'),i.get('rule_grade')) for i in b['item_comparisons']])
 def test_server_saves_comparison_payload_in_score(self):
  from server import score_samples
  x=self.segment();r=score_samples(list(enumerate(x)),640,480,'squats','front',True)
  self.assertEqual(len(r['method_comparison']['item_comparisons']),4);self.assertIn('trunk_balance',r['method_comparison']['feature_evidence'])
