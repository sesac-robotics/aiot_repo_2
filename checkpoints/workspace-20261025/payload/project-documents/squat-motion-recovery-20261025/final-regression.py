import sys,unittest,importlib.util
from pathlib import Path
O=Path(__file__).resolve().parent;A=O.parents[1]/'output/live-webcam-20261023';sys.path[:0]=[str(O),str(A)]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
module('live_observation',O/'live_observation_candidate.py')
m=module('movement',O/'movement_stage_candidate.py');sys.modules['movement_candidate']=m
c=module('checklist_comparison',O/'checklist_comparison_candidate.py');c.ROOT=A
names=['test_movement','test_live_observation','test_readiness_gate','test_exercise_integration','test_start_reservation','test_interrupted_session','test_item_observation','test_method_comparison','test_checklist_comparison','test_unified_assessment','test_score_display','test_front_candidate','test_rubric100','test_score_isolation']
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n)for n in names)
import test_stage
import test_motion_recovery as base
for cls in [base.MotionRecoveryTest,base.TimeAxisTest,test_stage.StageTest]:
 for name in unittest.defaultTestLoader.getTestCaseNames(cls):
  if cls==base.MotionRecoveryTest and name in ['test_short_knee_gap_recovers_without_imputation','test_coverage_uses_actual_missing_slots']:continue
  suite.addTest(cls(name))
r=unittest.TextTestRunner(verbosity=1).run(suite);sys.exit(not r.wasSuccessful())
