import unittest
from pathlib import Path
import numpy as np
from item_observation import observed_features, REQUIREMENTS

ROOT=Path(__file__).parent
DATA=ROOT.parent/'squat-item-observation-20261023'

class ItemObservationTests(unittest.TestCase):
    def sample(self):
        z=np.load(ROOT/'sessions/1792740109492452117/pose.npz')
        return z['image_xy'][(z['frame_numbers']>=7622)&(z['frame_numbers']<=7667)].copy()

    def test_head_and_ankles_are_not_requirements_and_do_not_change_features(self):
        x=self.sample();a=observed_features(x)
        x[:,:11]=np.nan;x[:,27:]=np.nan
        self.assertEqual(a,observed_features(x))
        for spec in REQUIREMENTS.values():
            self.assertFalse(set(spec['joints'])&set(list(range(11))+list(range(27,33))))

    def test_shoulder_missing_does_not_gate_knee(self):
        x=self.sample();x[:,[11,12]]=np.nan;f=observed_features(x)
        self.assertIn('knee_inward',f)
        self.assertNotIn('trunk_balance',f)

    def test_knee_missing_does_not_gate_body(self):
        x=self.sample();x[:,[25,26]]=np.nan;f=observed_features(x)
        self.assertNotIn('knee_inward',f)
        self.assertIn('trunk_balance',f)

    def test_off_image_points_are_not_observations(self):
        x=self.sample();x[:,25,1]=1000
        self.assertNotIn('knee_inward',observed_features(x,640,480))
        self.assertIn('trunk_balance',observed_features(x,640,480))

    def test_actual_head_cropped_source_recovers_knee_only(self):
        z=np.load(DATA/'observations/00050396/front.npz');x=z['image_xy'][z['frame_numbers']<=79]
        f=observed_features(x,int(z['width']),int(z['height']))
        self.assertIn('knee_inward',f)
        self.assertNotIn('trunk_balance',f)

    def test_temporary_initial_missing_not_entire_source_rejection(self):
        z=np.load(DATA/'observations/00232606/front.npz')
        f=observed_features(z['image_xy'],int(z['width']),int(z['height']))
        self.assertIn('knee_inward',f)

if __name__=='__main__':unittest.main()
