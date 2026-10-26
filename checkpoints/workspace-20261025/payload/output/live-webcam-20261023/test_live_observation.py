"""실제 오류 세션과 유효 입력의 검출/점수 관측 계약 회귀 검사."""
import unittest
from pathlib import Path
import numpy as np
from live_observation import mask_frame_border, movement_observation
from movement import MovementTracker

ROOT=Path(__file__).parent

class ObservationContractTest(unittest.TestCase):
    def test_mask_matches_deployed_scoring_boundary_and_preserves_raw(self):
        from server import RUNTIME_ROOT
        from app.scoring.three_items import mask_image_border
        z=np.load(ROOT/'sessions/1792846279497155109/pose.npz');x=z['image_xy'];original=x.copy()
        np.testing.assert_equal(mask_frame_border(x,640,480),mask_image_border(x,640,480))
        np.testing.assert_equal(x,original)
        # Exact 2% edges remain valid, coordinates beyond them do not.
        x=np.full((33,2),[320.,240.]);x[23]=[12.8,9.6];x[24]=[627.2,470.4]
        self.assertTrue(np.isfinite(mask_frame_border(x,640,480)).all())
        x[23,0]-=.01;self.assertTrue(np.isnan(mask_frame_border(x,640,480)[23]).all())

    def test_recorded_outside_hip_shoulder_cycle_is_not_counted(self):
        z=np.load(ROOT/'sessions/1792846279497155109/pose.npz');t=MovementTracker('shoulder','front')
        for tick,xy in zip(z['frame_numbers'],z['image_xy']):t.push(int(tick),xy,width=640,height=480)
        self.assertEqual(t.count,0)
        # The exact previously counted segment had both hips outside the image.
        segment=z['image_xy'][(z['frame_numbers']>=1545)&(z['frame_numbers']<=1583)]
        for xy in segment:
            o=movement_observation(xy,640,480,'shoulder','front')
            self.assertFalse(o['ready']);self.assertIn('왼쪽 골반',o['missing'])
            if np.isfinite(xy[23]).all():self.assertIn('왼쪽 골반',o['edge_joints'])

    def test_same_shoulder_motion_in_frame_still_counts_and_has_features(self):
        from server import RUNTIME_ROOT
        from app.scoring.three_items import describe
        z=np.load(ROOT/'sessions/1792846279497155109/pose.npz')
        seq=z['image_xy'][(z['frame_numbers']>=1525)&(z['frame_numbers']<=1590)].copy()
        # Geometry-only synthetic positive case: translate into frame and use
        # consecutive fully observed poses. Never alter the saved recording.
        seq[:,:,1]-=100
        required=[11,12,13,14,15,16,23,24]
        seq=seq[np.isfinite(seq[:,required]).all((1,2))]
        t=MovementTracker('shoulder','front');segments=[]
        for tick,xy in enumerate(seq):
            seg=t.push(tick,xy,width=640,height=480)
            if seg:segments.append(seg)
        self.assertEqual(t.count,1)
        x=np.array([xy for _,xy in segments[0]])
        notes=describe('shoulder',mask_frame_border(x,640,480),None)
        self.assertTrue(all(n['eligible'] for n in notes))
        from server import score_samples
        result=score_samples(segments[0],640,480,'shoulder','front')
        self.assertTrue(any(i['ai_grade'] is not None or i['rule_grade'] is not None for i in result['items']))

    def test_missing_slots_stay_in_coverage_denominator(self):
        from live_observation import segment_coverage
        x=np.full((33,2),[320.,240.])
        self.assertAlmostEqual(segment_coverage([(i,x) for i in [0,1,2,4,5,6,8]]),7/9)

    def test_sparse_in_frame_shoulder_cycle_is_held_not_counted(self):
        z=np.load(ROOT/'sessions/1792846279497155109/pose.npz')
        seq=z['image_xy'][(z['frame_numbers']>=1525)&(z['frame_numbers']<=1590)].copy()
        seq[:,:,1]-=100
        t=MovementTracker('shoulder','front')
        for tick,xy in enumerate(seq):t.push(tick,xy)
        self.assertEqual(t.count,0)
        self.assertLess(t.last_completion_coverage,.8)

    def test_front_squat_head_and_ankles_not_needed(self):
        from test_movement import squat
        t=MovementTracker('squats','front')
        for tick,x in enumerate([squat()]*10+[squat(True)]*10+[squat()]*10):
            x=x.copy();x[:11]=np.nan;x[27:]=np.nan
            t.push(tick,x)
        self.assertEqual(t.count,1)

    def test_pushup_guidance_names_arm_not_squat_knees(self):
        x=np.full((33,2),np.nan);x[[11,12,23,24,13,14,16]]=[320,240]
        o=movement_observation(x,640,480,'pushups','side')
        self.assertEqual(o['missing'],['왼쪽 손목']);self.assertNotIn('무릎',o['message'])

if __name__=='__main__':unittest.main()
