"""구도 준비 전 시작 차단 및 준비되면 실제 시작되는 HTTP 회귀검사."""
import json,time,threading,unittest,urllib.request,urllib.error
import numpy as np
from live_observation import ObservationReadyGate,require_start_ready
from test_shoulder_candidate import pose

class ReadyGateTests(unittest.TestCase):
    def test_six_consecutive_observations_and_missing_or_gap_reset(self):
        gate=ObservationReadyGate()
        for tick in range(5):self.assertFalse(gate.update(tick,pose(),640,480)['shoulder']['front']['start_ready'])
        self.assertTrue(gate.update(5,pose(),640,480)['shoulder']['front']['start_ready'])
        self.assertFalse(gate.update(7,pose(),640,480)['shoulder']['front']['start_ready'])
        x=pose();x[23]=np.nan
        o=gate.update(8,x,640,480)['shoulder']['front']
        self.assertEqual(o['ready_frames'],0);self.assertIn('왼쪽 골반',o['message'])

    def test_all_saved_missing_hip_frames_never_enable_start(self):
        from pathlib import Path
        z=np.load(Path(__file__).parent/'sessions/1792848383011095301/pose.npz');gate=ObservationReadyGate()
        for tick,xy in zip(z['frame_numbers'],z['image_xy']):
            self.assertFalse(gate.update(int(tick),xy,640,480)['shoulder']['front']['start_ready'])

    def test_stale_or_wrong_exercise_cannot_start(self):
        gate=ObservationReadyGate()
        for tick in range(6):o=gate.update(tick,pose(),640,480)
        require_start_ready({'camera_at':100.,'observation_options':o},'shoulder','front',100.2)
        with self.assertRaises(ValueError):require_start_ready({'camera_at':100.,'observation_options':o},'shoulder','front',102.)
        self.assertFalse(o['ytw']['front']['start_ready'])
        self.assertFalse(o['shoulder']['side']['start_ready'])

    def test_http_begin_blocks_without_mutation_then_allows_ready(self):
        import server
        old=dict(server.STATE);old_tracker=server.TRACKER
        http=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
        def begin():
            req=urllib.request.Request(f'http://127.0.0.1:{http.server_port}/begin',data=b'{"exercise":"shoulder","view":"front"}',headers={'Content-Type':'application/json'},method='POST')
            return urllib.request.urlopen(req)
        try:
            gate=ObservationReadyGate();x=pose();x[23:25]=np.nan
            options=gate.update(0,x,640,480)
            with server.LOCK:server.STATE.update(collecting=False,finalizing=False,jpeg=b'test',camera_at=time.monotonic(),update_available=False,observation_options=options,generation=12)
            with self.assertRaises(urllib.error.HTTPError) as e:begin()
            self.assertEqual(e.exception.code,400)
            self.assertIn('측정 준비 대기',json.loads(e.exception.read())['error'])
            self.assertFalse(server.STATE['collecting']);self.assertEqual(server.STATE['generation'],12)
            for tick in range(1,7):options=gate.update(tick,pose(),640,480)
            with server.LOCK:server.STATE.update(camera_at=time.monotonic(),observation_options=options)
            with begin() as response:self.assertEqual(response.status,200)
            self.assertTrue(server.STATE['collecting']);self.assertEqual(server.STATE['generation'],13)
        finally:
            http.shutdown();http.server_close();thread.join()
            with server.LOCK:server.STATE.clear();server.STATE.update(old);server.TRACKER=old_tracker

if __name__=='__main__':unittest.main()
