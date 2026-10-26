"""중단 자료는 정상 세션/실패 정답으로 바꾸지 않고 별도로 보존한다."""
import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import server

class InterruptedSessionTests(unittest.TestCase):
    def setUp(self):self.old=dict(server.STATE)
    def tearDown(self):
        with server.LOCK:server.STATE.clear();server.STATE.update(self.old)
    def active(self):
        points=np.full((33,2),np.nan);points[[11,12,23,24]]=[320.,240.]
        with server.LOCK:server.STATE.update(collecting=True,samples=[(i,points.copy()) for i in range(12)],width=640,height=480,repetitions=[],generation=99)
    def test_camera_disconnect_saves_raw_without_counting_failure(self):
        self.active()
        with tempfile.TemporaryDirectory() as temp,patch.object(server,'__file__',str(Path(temp)/'server.py')):
            folder=server.preserve_interrupted_session('camera disconnected')
            self.assertIn('interrupted-sessions',str(folder))
            self.assertFalse(server.STATE['collecting'])
            z=np.load(folder/'pose.npz');self.assertEqual(len(z['frame_numbers']),12)
            s=json.loads((folder/'interrupted-state.json').read_text())
            self.assertTrue(s['interrupted']);self.assertEqual(s['repetitions'],[])
            self.assertFalse((folder/'final-state.json').exists())
            self.assertIsNone(server.preserve_interrupted_session('second disconnect'))
    def test_signal_saves_then_exits_without_auto_restart(self):
        self.active()
        with tempfile.TemporaryDirectory() as temp,patch.object(server,'__file__',str(Path(temp)/'server.py')):
            with self.assertRaises(SystemExit):server.stop_on_signal(15,None)
            self.assertFalse(server.STATE['collecting'])
            folder=Path(server.STATE['last_interrupted_session_folder'])
            self.assertTrue((folder/'pose.npz').exists())
    def test_idle_does_not_create_fake_session(self):
        with server.LOCK:server.STATE['collecting']=False
        with tempfile.TemporaryDirectory() as temp,patch.object(server,'__file__',str(Path(temp)/'server.py')):
            self.assertIsNone(server.preserve_interrupted_session('idle'))
            self.assertEqual(list(Path(temp).iterdir()),[])
