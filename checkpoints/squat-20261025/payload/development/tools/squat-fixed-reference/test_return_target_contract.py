import copy
import unittest
from return_target_contract import ANATOMICAL, READY_RELATIVE, SCHEMAS, assess, compatible_with_head

class ReturnTargetContractTests(unittest.TestCase):
    def setUp(self):
        self.source=dict(id='example',source_sha256='a'*64,person='example_person',
            split='train',view='front',frame_count=120)
        self.a=dict(self.source,exercise='shoulder',posture='seated',criterion=ANATOMICAL,
            protocol='shoulder_press',laterality='bilateral',
            label_origin='direct_source_visual_review',review_status='reviewed_AI_assisted',
            grade=1,evidence_sha256={'test_evidence':'b'*64},scope='selected_event_only',
            event_source_sha256='a'*64,peak_frames=[0,8],turn_frames=[20,26],
            next_ascent_frames=[30,36],interrupted_or_clipped=False,
            actual_pixels_visible=True,rubric='anatomical_rack_range_v1',closure='observed_next_ascent')
        self.obs=dict(status='passed',source_sha256='a'*64,feature_schema=SCHEMAS[ANATOMICAL],
            target_person_tracking_QA=True,actual_10Hz_slots_80pct_and_2pct_border=True,
            no_interpolation_or_tracking_break=True,posture='seated',view='front',
            event_frames=[[0,8],[20,26],[30,36]])
    def check(self,a=None,source=None,obs=None):
        return assess(a or self.a,source or self.source,observation=obs or self.obs,
            registered_schemas=[SCHEMAS[ANATOMICAL],SCHEMAS[READY_RELATIVE]])
    def test_anatomical_visual_label_does_not_require_initial_rack(self):
        r=self.check();self.assertTrue(r['visual_label_eligible']);self.assertTrue(r['training_eligible'])
        self.assertFalse(r['grade_created']);self.assertIsNone(r['score']);self.assertFalse(r['production_approved'])
    def test_relative_target_still_requires_causal_ready(self):
        self.a['criterion']=READY_RELATIVE
        self.assertIn('causal_same_source_ready_reference_missing',self.check()['visual_label_holds'])
    def test_later_ready_cannot_be_backdated(self):
        self.a.update(criterion=READY_RELATIVE,ready_frames=[50,56],ready_source_sha256='a'*64)
        self.assertFalse(self.check()['visual_label_eligible'])
    def test_other_clip_ready_not_allowed(self):
        self.a.update(criterion=READY_RELATIVE,peak_frames=[10,15],ready_frames=[0,5],ready_source_sha256='c'*64)
        self.assertFalse(self.check()['visual_label_eligible'])
    def test_same_source_preceding_ready_can_supervise_relative_target(self):
        self.a.update(criterion=READY_RELATIVE,peak_frames=[10,15],ready_frames=[0,5],ready_source_sha256='a'*64)
        self.obs.update(feature_schema=SCHEMAS[READY_RELATIVE],event_frames=[[10,15],[20,26],[30,36]])
        self.assertTrue(self.check()['training_eligible'])
    def test_validation_is_never_train(self):
        self.source['split']=self.a['split']='validation';r=self.check()
        self.assertTrue(r['visual_label_eligible']);self.assertTrue(r['evaluation_input_eligible'])
        self.assertFalse(r['training_eligible'])
    def test_proposed_role_change_is_rejected(self):
        self.source['split']='test'
        self.assertIn('canonical_source_mismatch:split',self.check()['visual_label_holds'])
    def test_end_without_following_ascent_is_not_partial_return(self):
        self.a['next_ascent_frames']=None
        self.assertFalse(self.check()['visual_label_eligible'])
    def test_normal_observed_rack_can_close_without_next_ascent(self):
        self.a.update(grade=2,closure='observed_rack',rack_visible_stable=True,next_ascent_frames=None)
        self.obs['event_frames']=[[0,8],[20,26],None]
        self.assertTrue(self.check()['training_eligible'])
    def test_rack_flag_cannot_close_partial_error_without_next_ascent(self):
        self.a.update(grade=1,closure='observed_rack',rack_visible_stable=True,next_ascent_frames=None)
        self.assertFalse(self.check()['visual_label_eligible'])
    def test_rack_closure_requires_actual_observed_rack(self):
        self.a.update(grade=2,closure='observed_rack',next_ascent_frames=None)
        self.assertFalse(self.check()['visual_label_eligible'])
    def test_return_samples_cannot_precede_peak(self):
        self.a['turn_frames']=[3,6]
        self.assertFalse(self.check()['visual_label_eligible'])
    def test_interruption_does_not_become_zero_grade(self):
        self.a.update(interrupted_or_clipped=True,grade=0)
        self.assertIsNone(self.check()['manual_grade'])
    def test_provider_or_rule_grade_not_a_visual_label(self):
        for origin in ['provider_bad','rule_prediction','AI_probability']:
            with self.subTest(origin=origin):
                self.a['label_origin']=origin;self.assertFalse(self.check()['visual_label_eligible'])
    def test_near_arm_only_cannot_supervise_bilateral_head(self):
        self.a['laterality']='near_arm_only';self.assertFalse(self.check()['visual_label_eligible'])
    def test_nonpress_does_not_become_partial_press(self):
        self.a['protocol']='non_press_arm_extension';self.assertFalse(self.check()['visual_label_eligible'])
    def test_tracking_mistake_holds_features_not_visual_grade(self):
        self.obs['target_person_tracking_QA']=False;r=self.check()
        self.assertTrue(r['visual_label_eligible']);self.assertFalse(r['training_eligible'])
    def test_feature_schema_mismatch_blocks_fit(self):
        self.obs['feature_schema']=SCHEMAS[READY_RELATIVE];self.assertFalse(self.check()['training_eligible'])
    def test_bank_posture_cannot_change(self):
        self.obs['posture']='standing';self.assertFalse(self.check()['training_eligible'])
    def test_actual_time_coverage_required(self):
        self.obs['actual_10Hz_slots_80pct_and_2pct_border']=False
        self.assertFalse(self.check()['training_eligible'])
    def test_unregistered_extractor_not_fit(self):
        r=assess(self.a,self.source,observation=self.obs)
        self.assertTrue(r['visual_label_eligible']);self.assertFalse(r['training_eligible'])
    def test_original_event_alignment_required(self):
        self.obs['event_frames']=[[0,8],[40,46],[50,56]]
        self.assertFalse(self.check()['training_eligible'])
    def test_existing_head_is_not_compatible_with_anatomical_target(self):
        r=self.check();head=dict(criterion=READY_RELATIVE,feature_schema=SCHEMAS[READY_RELATIVE],bank=r['bank'])
        self.assertFalse(compatible_with_head(r,head))
    def test_no_input_mutation(self):
        before=copy.deepcopy((self.a,self.source,self.obs));self.check()
        self.assertEqual(before,(self.a,self.source,self.obs))

if __name__=='__main__':unittest.main()
