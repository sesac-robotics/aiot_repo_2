import sys,unittest
from pathlib import Path
O=Path(__file__).resolve().parent;A=O.parents[1]/'output/live-webcam-20261023';sys.path[:0]=[str(O),str(A)]
import movement_candidate
sys.modules['movement']=movement_candidate
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n)for n in ['test_transition','test_movement','test_live_observation','test_readiness_gate','test_exercise_integration','test_start_reservation','test_interrupted_session','test_shoulder_candidate'])
r=unittest.TextTestRunner(verbosity=1).run(suite);sys.exit(not r.wasSuccessful())
