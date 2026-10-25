import unittest
import numpy as np
from features import extract,with_reference_pose

class TerminalWindow(unittest.TestCase):
    def sequence(self,curve):
        x=np.full((len(curve),33,2),np.nan)
        for k,v in enumerate(curve):
            x[k,[11,12]]=[[100,100+100*v],[110,100+100*v]]
            x[k,[23,24]]=[[100,200+100*v],[110,200+100*v]]
        return x
    def test_return_before_next_descent_not_lost(self):
        curve=np.r_[np.linspace(0,.8,15),np.linspace(.8,0,15),np.zeros(4),np.linspace(0,.5,12)]
        f=extract(self.sequence(curve),dict(hip_y=200,shoulder_y=100,center_x=105,scale=100,torso_x=0))
        self.assertIsNotNone(f);self.assertLess(f['x'][0],.03);self.assertLess(f['return_index'],35)
    def test_partial_return_keeps_actual_residual(self):
        curve=np.r_[np.linspace(0,.8,15),np.linspace(.8,.25,15),np.repeat(.25,4),np.linspace(.25,.7,8)]
        f=extract(self.sequence(curve),dict(hip_y=200,shoulder_y=100,center_x=105,scale=100,torso_x=0))
        self.assertAlmostEqual(f['x'][0],.25)
    def test_deeper_clipped_next_descent_does_not_erase_return(self):
        curve=np.r_[np.linspace(0,.5,15),np.linspace(.5,0,15),np.zeros(4),np.linspace(0,.9,15)]
        f=extract(self.sequence(curve),dict(hip_y=200,shoulder_y=100,center_x=105,scale=100,torso_x=0))
        self.assertIsNotNone(f);self.assertLess(f['return_index'],35);self.assertLess(f['x'][0],.03)
    def test_missing_required_joints_held(self):
        x=self.sequence(np.r_[np.linspace(0,.8,15),np.linspace(.8,0,15)]);x[:,12]=np.nan
        self.assertIsNone(extract(x,dict(hip_y=200,shoulder_y=100,center_x=105,scale=100,torso_x=0)))
    def test_no_motion_held(self):
        self.assertIsNone(extract(self.sequence(np.zeros(30)),dict(hip_y=200,shoulder_y=100,center_x=105,scale=100,torso_x=0)))
    def test_reference_pose_requires_observed_chain(self):
        x=self.sequence(np.r_[np.linspace(0,.8,15),np.linspace(.8,0,15)]);base=extract(x,dict(hip_y=200,shoulder_y=100,center_x=105,scale=100,torso_x=0))
        self.assertIsNone(with_reference_pose(x,x[:3],base))
    def test_reference_pose_keeps_vertical_height(self):
        x=self.sequence(np.r_[np.linspace(0,.8,15),np.linspace(.8,.25,15)]);x[:,[25,26]]=[[100,300],[110,300]];x[:,[27,28]]=[[100,400],[110,400]]
        ref=self.sequence(np.zeros(3));ref[:,[25,26]]=[[100,300],[110,300]];ref[:,[27,28]]=[[100,400],[110,400]]
        base=extract(x,dict(hip_y=200,shoulder_y=100,center_x=105,scale=100,torso_x=0));f=with_reference_pose(x,ref,base)
        self.assertIsNotNone(f);self.assertEqual(len(f['x']),15);self.assertGreater(f['x'][0],.2)

if __name__=='__main__':unittest.main()
