import sys
import unittest
from pathlib import Path
import numpy as np

B=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'tools/integrated-five')]
from team_features import describe, height_features
import side_height

def sequence(residual=0):
    x=np.full((40,33,2),np.nan)
    drop=np.r_[np.linspace(0,.6,20),np.linspace(.6,residual,20)]
    for k,d in enumerate(drop):
        x[k,[11,12]]=[[100,100+100*d],[110,100+100*d]]
        x[k,[23,24]]=[[100,200+100*d],[110,200+100*d]]
        x[k,[27,28]]=[[100,350],[110,350]]
    return x

class SideReturn(unittest.TestCase):
    def ref(self,x):return side_height.reference(x[:3],height_features)
    def measure(self,x,**kw):
        return describe('squats',x,'side',4,**kw)[1]
    def test_missing_verified_reference_is_held(self):
        self.assertFalse(self.measure(sequence())['eligible'])
    def test_future_reference_is_held(self):
        self.assertFalse(self.measure(sequence(),baseline=height_features.reference(sequence()),baseline_verified=True,baseline_end=5,segment_start=4)['eligible'])
    def test_no_future_or_automatic_reference(self):
        self.assertFalse(self.measure(sequence(),baseline_verified=True,baseline_end=0,segment_start=1)['eligible'])
    def test_verified_side_matches_existing_physical_geometry(self):
        x=sequence();ref=self.ref(x)
        n=self.measure(x,baseline=ref,baseline_verified=True,baseline_end=0,segment_start=1)
        self.assertTrue(n['eligible'])
        self.assertEqual(n['feature'],height_features.extract(x,ref)['x'])
        self.assertEqual(n['bank'],'height_v4')
    def test_partial_return_retains_height_difference(self):
        ref=self.ref(sequence())
        good=self.measure(sequence(),baseline=ref,baseline_verified=True,baseline_end=0,segment_start=1)
        bad=self.measure(sequence(.3),baseline=ref,baseline_verified=True,baseline_end=0,segment_start=1)
        self.assertGreater(bad['feature'][0],good['feature'][0]+.2)
    def test_missing_joints_not_imputed(self):
        x=sequence();ref=self.ref(x);x[:,12]=np.nan
        self.assertFalse(self.measure(x,baseline=ref,baseline_verified=True,baseline_end=0,segment_start=1)['eligible'])
    def test_front_output_unchanged(self):
        x=sequence();ref=height_features.reference(x)
        n=describe('squats',x,'front',4,ref)[1]
        self.assertEqual(n['feature'],height_features.extract(x,ref)['x'])
    def test_no_movement_is_held(self):
        x=sequence();ref=self.ref(x);x[:]=x[0]
        self.assertFalse(self.measure(x,baseline=ref,baseline_verified=True,baseline_end=0,segment_start=1)['eligible'])
    def test_expected_horizontal_hip_travel_is_supported(self):
        x=sequence();ref=self.ref(x);x[:,[11,12,23,24],0]+=np.sin(np.linspace(0,np.pi,40))[:,None]*60
        self.assertTrue(self.measure(x,baseline=ref,baseline_verified=True,baseline_end=0,segment_start=1)['eligible'])
    def test_image_motion_with_feet_is_held(self):
        x=sequence();ref=self.ref(x);x[20:,:,0]+=40
        self.assertFalse(self.measure(x,baseline=ref,baseline_verified=True,baseline_end=0,segment_start=1)['eligible'])
    def test_missing_feet_not_imputed(self):
        x=sequence();ref=self.ref(x);x[:,27]=np.nan
        x[20:,[11,12,23,24],0]+=60
        self.assertFalse(self.measure(x,baseline=ref,baseline_verified=True,baseline_end=0,segment_start=1)['eligible'])
    def test_hidden_feet_do_not_hide_observed_return_height(self):
        x=sequence();ref=self.ref(x);x[:,[27,28]]=np.nan
        x[:,[11,12,23,24],0]+=np.sin(np.linspace(0,np.pi,40))[:,None]*60
        self.assertTrue(self.measure(x,baseline=ref,baseline_verified=True,baseline_end=0,segment_start=1)['eligible'])
    def test_hinge_attempt_does_not_require_correct_descent(self):
        x=sequence();x[:,[23,24],1]=200
        ref=self.ref(x)
        note=self.measure(x,baseline=ref,baseline_verified=True,baseline_end=0,segment_start=1)
        self.assertTrue(note['eligible'])
        self.assertEqual(note['phase_source'],'observed_shoulder_excursion_with_insufficient_hip_descent')

if __name__=='__main__':unittest.main()
