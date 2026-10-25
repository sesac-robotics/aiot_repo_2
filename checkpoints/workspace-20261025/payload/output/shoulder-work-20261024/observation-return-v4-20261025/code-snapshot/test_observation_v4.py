import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'output/live-webcam-20261023'))
from test_shoulder_candidate import pose
from observation_features_v4 import extract_return,extract_body
from score_items_v4 import score,SCHEMAS,TARGETS
class BombModel:
    def predict(self,x):raise AssertionError('unsupported model must never be called')
class Tests(unittest.TestCase):
    def seq(self):return [pose().copy() for _ in range(6)]+[pose(True).copy() for _ in range(4)]+[pose().copy() for _ in range(3)]
    def samples(self,s=None):return list(enumerate(self.seq() if s is None else s))
    def test_new_return_uses_corrected_boundary(self):
        r=extract_return(self.samples());self.assertEqual(r['return_ticks'],[10,11,12]);self.assertNotIn('trunk_features',r)
    def test_body_survives_all_arm_missing(self):
        s=self.seq()
        for p in s:p[[13,14,15,16]]=np.nan
        self.assertEqual(extract_return(self.samples(s))['status'],'held')
        b=extract_body(self.samples(s),[0,12],True)
        self.assertEqual(b['status'],'features_ready');self.assertIsNone(b['grade'])
    def test_sitting_or_idle_not_automatically_exercise(self):
        self.assertEqual(extract_body(self.samples(),[0,12])['reason'],'exercise_window_not_confirmed')
    def test_window_missing_must_not_shrink(self):
        self.assertEqual(extract_body(self.samples(),[-1,12],True)['reason'],'window_not_fully_supplied')
    def test_baseline_not_shifted_over_missing(self):
        s=self.seq();s[0][11]=np.nan
        self.assertEqual(extract_body(self.samples(s),[0,12],True)['reason'],'window_initial_reference_unobservable')
    def test_collapsed_hip_is_uncertainty_not_grade(self):
        s=self.seq();s[7][24]=s[7][23]
        for f in [extract_return(self.samples(s)),extract_body(self.samples(s),[0,12],True)]:
            self.assertEqual(f['reason'],'tracking_geometry_uncertain');self.assertNotIn('grade',f)
    def test_actual_lean_is_not_quality_failure(self):
        s=self.seq()
        for i,p in enumerate(s):
            # Rigid 25-degree body rotation keeps topology; tilt must be measured.
            theta=np.deg2rad(25 if i>5 else 0);R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            p[:]=(p-[320,330])@R.T+[320,330]
        b=extract_body(self.samples(s),[0,12],True)
        self.assertEqual(b['status'],'features_ready');self.assertGreater(b['features'][4],20)
    def test_missing_slots_not_compacted(self):
        s=self.samples();s=[(t*2,p) for t,p in s]
        self.assertEqual(extract_body(s,[0,24],True)['reason'],'insufficient_actual_slot_observation')
    def test_missing_head_and_ankles_irrelevant(self):
        self.assertEqual(extract_body(self.samples(),[0,12],True)['status'],'features_ready')
    def test_side_not_subject_to_frontal_quality(self):
        self.assertEqual(extract_return(self.samples(),view='side')['reason'],'frontal_geometry_required')
    def test_no_model_no_filled_points(self):
        r=score(self.samples());self.assertFalse(r['all_items_supported']);self.assertTrue(all(x['points'] is None for x in r['items']))
    def test_normal_only_old_model_never_called(self):
        a=dict(model=BombModel(),training_grade_counts={'2':9},feature_schema='old',criterion='old')
        r=score(self.samples(),{k:a for k in SCHEMAS})
        for x in r['items']:self.assertIn('normal_only_or_missing_normal_training',x['reasons']);self.assertIn('feature_schema_mismatch',x['reasons'])
    def test_no_ready_body_window_remains_separate(self):
        s=self.samples([pose(True).copy() for _ in range(13)])
        r=score(s,trunk_window=[0,12],exercise_confirmed=True)
        self.assertEqual(r['features']['lower_return']['reason'],'no_observed_ready_reference')
        self.assertEqual(r['features']['frontal_trunk_hold']['status'],'features_ready')
        self.assertIsNone(r['items'][1]['points'])
    def test_typed_scoring_gate_with_no_provenance(self):
        models={k:dict(model=BombModel(),training_grade_counts={'1':1,'2':9},feature_schema=SCHEMAS[k],criterion=TARGETS[k],view='front',pose_context='standing',development_gate={'passed':True},score_prediction_enabled=True) for k in SCHEMAS}
        r=score(self.samples(),models,pose_context='standing')
        for x in r['items']:self.assertEqual(x['reasons'],['missing_training_provenance'])
    def test_later_bad_tracking_does_not_reject_completed_first_attempt(self):
        s=self.seq()+[pose().copy() for _ in range(5)]
        s[-1][24]=s[-1][23]
        r=extract_return(self.samples(s))
        self.assertEqual(r['status'],'features_ready');self.assertEqual(r['return_ticks'],[10,11,12])
    def test_wrong_pose_bank_model_never_called(self):
        models={k:dict(model=BombModel(),training_grade_counts={'1':1,'2':9},feature_schema=SCHEMAS[k],criterion=TARGETS[k],view='front',pose_context='standing',development_gate={'passed':True},score_prediction_enabled=True,training_ids=['example'],training_source_hashes=['example']) for k in SCHEMAS}
        r=score(self.samples(),models,pose_context='seated')
        for x in r['items']:self.assertEqual(x['reasons'],['pose_context_mismatch'])
    def test_noninteger_and_duplicate_ticks_rejected(self):
        for s in [[(0,pose()),(0,pose())],[(0.1,pose()),(1.1,pose())]]:
            with self.assertRaises(ValueError):extract_return(s)
if __name__=='__main__':unittest.main()
