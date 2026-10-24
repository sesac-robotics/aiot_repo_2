import sys,unittest
from pathlib import Path
O=Path(__file__).resolve().parent;A=O.parents[1]/'output/live-webcam-20261023';sys.path[:0]=[str(O),str(A)]
import movement_stage_candidate as movement_candidate
sys.modules['movement']=movement_candidate
sys.modules['movement_candidate']=movement_candidate
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n)for n in ['test_movement','test_live_observation','test_readiness_gate','test_exercise_integration','test_start_reservation','test_interrupted_session','test_shoulder_candidate'])
import test_stage
import test_motion_recovery as base
for cls in [base.MotionRecoveryTest,base.TimeAxisTest,test_stage.StageTest]:
    for name in unittest.defaultTestLoader.getTestCaseNames(cls):
        if cls==base.MotionRecoveryTest and name in ['test_short_knee_gap_recovers_without_imputation','test_coverage_uses_actual_missing_slots']:continue
        suite.addTest(cls(name))
r=unittest.TextTestRunner(verbosity=1).run(suite);sys.exit(not r.wasSuccessful())
