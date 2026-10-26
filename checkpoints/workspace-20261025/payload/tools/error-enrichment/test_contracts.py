import unittest
import numpy as np
from acceptance import assess
from selection import choose
from models import ScaledRegressor
from pose_features import relative_knee_gap,relative_trunk
class Contracts(unittest.TestCase):
 def test_each_person_has_normal_and_error_on_both_sides(self):
  import json
  from split_guard import POLICY
  p=json.loads(POLICY.read_text())
  for person in p['people']:
   for quality in ['normal','error']:
    counts=person['classes'][quality]
    self.assertGreater(counts['train'],0);self.assertGreater(counts['test'],0)
 def test_every_exercise_has_normal_and_error_test_examples(self):
  import json
  from pathlib import Path
  from split_guard import ROOT
  rows=json.loads((ROOT/'output/per-person-stratified-80-20-20261024/clip-assignments.frozen.json').read_text())
  expected={(e,q) for e in ['squats','pushups','shoulder','plank','ytw'] for q in ['normal','error']}
  actual={(r['exercise'],r['quality_class']) for r in rows if r['split']=='test'}
  self.assertTrue(expected<=actual)
 def test_holdout_clip_cannot_train_even_if_split_changed(self):
  import json
  from split_guard import POLICY,assert_training_allowed
  clip=json.loads(POLICY.read_text())['protected_clip_ids'][0]
  with self.assertRaises(ValueError):assert_training_allowed([dict(id=clip,person='alias',split='train')])
 def test_holdout_source_cannot_train_under_new_person_name(self):
  import json
  from split_guard import POLICY,assert_training_allowed
  source=json.loads(POLICY.read_text())['protected_source_sha256'][0]
  with self.assertRaises(ValueError):assert_training_allowed([dict(id='renamed',person='alias',split='train',source_sha256=source)])
 def test_validation_row_cannot_enter_fit(self):
  from split_guard import assert_training_allowed
  with self.assertRaises(ValueError):assert_training_allowed([dict(id='new',person='new_person',split='validation')])
 def test_all_four_people_may_use_their_training_portion(self):
  from split_guard import assert_training_allowed
  assert_training_allowed([dict(id='allowed_'+p,person='r2d2_'+p,split='train') for p in ['P01','P02','P03','P04']])
 def test_no_validation_cannot_pass(self):
  self.assertFalse(assess([],[],{'balanced_group_MAE':1})['passed'])
 def test_normal_only_validation_cannot_pass(self):
  ps=[dict(target=2,prediction=2,group=str(i)) for i in range(8)]
  self.assertFalse(assess(ps,ps,{'balanced_group_MAE':1})['passed'])
 def test_repeated_one_person_error_not_independent(self):
  ps=[dict(target=1,prediction=1,group='same') for _ in range(20)]+[dict(target=2,prediction=2,group=str(i)) for i in range(5)]
  self.assertFalse(assess(ps,ps,{'balanced_group_MAE':1})['passed'])
 def test_selection_does_not_reward_missing_all_errors(self):
  base=dict(name='all_normal',metrics=dict(error_recall=0.,normal_false_alarm=0.,balanced_group_MAE=.2))
  useful=dict(name='detects_both',metrics=dict(error_recall=.8,normal_false_alarm=.1,balanced_group_MAE=.3))
  self.assertEqual(choose([base,useful])['name'],'detects_both')
 def test_unseen_severe_grade_not_invented(self):
  X=np.array([[0],[.1],[1],[1.1]]);y=np.array([2,2,1,1]);m=ScaledRegressor('grade1').fit(X,y)
  self.assertTrue(set(m.predict([[0],[.5],[5]]))<={1,2})
 def test_one_class_fit_is_fold_fallback_only(self):
  m=ScaledRegressor('grade1').fit([[0],[1]],[2,2]);self.assertEqual(m.predict([[10]])[0],2)
 def test_missing_lower_body_not_knee_success(self):
  x=np.full((15,33,2),np.nan);self.assertIsNone(relative_knee_gap(x))
 def test_missing_torso_not_stability_success(self):
  x=np.full((15,33,2),np.nan);self.assertIsNone(relative_trunk(x))
 def test_eligibility_does_not_change_target_grades(self):
  import json
  from common import O
  original={r['id']:r for r in json.loads((O/'score-inputs-no-transfer.frozen.json').read_text())}
  new=json.loads((O/'score-inputs.frozen.json').read_text())
  revisions={r['id']:r for r in json.loads((O/'label-revisions.frozen.json').read_text())['revisions']}
  for r in new:
   expected=list(original[r['id']]['levels'])
   if r['id'] in revisions:
    for i in revisions[r['id']]['affected_item_indices']:
     if i<len(expected):expected[i]=None
   self.assertEqual(r['levels'][:len(expected)],expected)
 def test_regression_does_not_claim_untrained_severe_return_grade(self):
  from train_scores import fit,predictions
  rs=[dict(id=str(i),exercise='squats',index=1,x=[x],target=y,person=str(i),group=str(i),family='test') for i,(x,y) in enumerate([(0,2),(.1,2),(1,1),(1.1,1)])]
  m=fit(rs,'ridge1');probe=dict(rs[0],x=[100]);self.assertGreaterEqual(predictions(m,[probe])[0]['prediction'],1)
if __name__=='__main__':unittest.main()
