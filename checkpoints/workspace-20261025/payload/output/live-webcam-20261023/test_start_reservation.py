import unittest
from start_reservation import StartReservation

class ReservationTests(unittest.TestCase):
    def test_unarmed_never_starts(self):
        r=StartReservation();self.assertIsNone(r.step(5,True));self.assertEqual(r.status()['state'],'idle')
    def test_explicit_arm_ready_countdown_starts_once(self):
        r=StartReservation();r.arm('shoulder','front',0,None)
        self.assertIsNone(r.step(.5,False));self.assertIsNone(r.step(1,True))
        self.assertEqual(r.status()['remaining'],3)
        r.heartbeat(r.status()['request_id'],3)
        self.assertIsNone(r.step(3,True));self.assertEqual(r.status()['remaining'],1)
        self.assertEqual(r.step(4,True)['exercise'],'shoulder')
        self.assertIsNone(r.step(5,True))
    def test_visibility_or_pose_loss_resets_countdown(self):
        r=StartReservation();r.arm('shoulder','front',0,None);r.step(0,True);r.step(2,False)
        r.heartbeat(r.status()['request_id'],3)
        self.assertIsNone(r.step(3,True));self.assertEqual(r.status()['remaining'],3)
    def test_cancel_or_disconnect_does_not_restart(self):
        for reason in ['cancel','exercise_changed','camera_disconnected','server_reloaded']:
            r=StartReservation();r.arm('shoulder','front',0,None);r.cancel(reason)
            self.assertIsNone(r.step(5,True));self.assertIsNone(r.status()['request_id'])
        r=StartReservation();r.arm('shoulder','front',0,None);self.assertIsNone(r.step(4.1,True))
        self.assertEqual(r.state,'cancelled')
    def test_expiry_and_stale_heartbeat_token(self):
        r=StartReservation();r.arm('shoulder','front',0,None);old=r.status()['request_id']
        r.arm('squats','front',1,None);r.heartbeat(old,5)
        self.assertIsNone(r.step(5.1,True));self.assertEqual(r.state,'cancelled')
        r.arm('shoulder','front',0,None);r.heartbeat(r.status()['request_id'],181)
        self.assertIsNone(r.step(181,True));self.assertEqual(r.state,'cancelled')

class CameraReservationIntegrationTests(unittest.TestCase):
    def test_actual_shoulder_lower_probe_and_countdown_start_without_click(self):
        import server,time
        from unittest.mock import patch
        from live_observation import ObservationReadyGate
        from test_shoulder_candidate import pose
        state=dict(server.STATE);old_request=server.START_REQUEST;old_tracker=server.TRACKER;old_coach=server.COACH
        base=time.monotonic()
        try:
            server.START_REQUEST=StartReservation()
            server.START_REQUEST.arm('shoulder','front',base,server.make_tracker('shoulder'))
            gate=ObservationReadyGate()
            with server.LOCK:server.STATE.update(collecting=False,finalizing=False,jpeg=b'preview',update_available=False,generation=44)
            for tick in range(36):
                now=base+tick/10
                with server.LOCK,patch.object(server.time,'monotonic',return_value=now):
                    server.STATE.update(camera_at=now,observation_options=gate.update(tick,pose(),640,480))
                    server.START_REQUEST.heartbeat(server.START_REQUEST.status()['request_id'],now)
                    server.advance_start_locked(tick,pose(),640,480,now)
                if tick<35:self.assertFalse(server.STATE['collecting'])
            self.assertTrue(server.STATE['collecting']);self.assertEqual(server.STATE['generation'],45)
            self.assertEqual(server.STATE['exercise'],'shoulder')
            self.assertIsNone(server.START_REQUEST.status()['request_id'])
        finally:
            with server.LOCK:
                server.STATE.clear();server.STATE.update(state)
                server.START_REQUEST=old_request;server.TRACKER=old_tracker;server.COACH=old_coach

class ReservationHttpTests(unittest.TestCase):
    def test_arm_without_pose_then_cancel_or_end_never_collects(self):
        import server,time,json,threading,urllib.request
        from live_observation import ObservationReadyGate
        from test_shoulder_candidate import pose
        state=dict(server.STATE);old_request=server.START_REQUEST
        http=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        worker=threading.Thread(target=http.serve_forever,daemon=True);worker.start()
        def post(path,data={}):
            req=urllib.request.Request(f'http://127.0.0.1:{http.server_port}'+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'},method='POST')
            with urllib.request.urlopen(req) as response:return response.status
        try:
            import numpy as np
            x=pose();x[23:25]=np.nan
            options=ObservationReadyGate().update(0,x,640,480)
            server.START_REQUEST=StartReservation()
            with server.LOCK:server.STATE.update(collecting=False,finalizing=False,jpeg=b'preview',camera_at=time.monotonic(),update_available=False,observation_options=options)
            self.assertEqual(post('/arm-start',{'exercise':'shoulder','view':'front'}),200)
            self.assertFalse(server.STATE['collecting']);self.assertIsNotNone(server.START_REQUEST.status()['request_id'])
            post('/cancel-start');self.assertIsNone(server.START_REQUEST.status()['request_id'])
            post('/arm-start',{'exercise':'shoulder','view':'front'});post('/end')
            self.assertFalse(server.STATE['collecting']);self.assertIsNone(server.START_REQUEST.status()['request_id'])
            post('/arm-start',{'exercise':'shoulder','view':'front'})
            server.preserve_interrupted_session('camera lost before capture')
            self.assertIsNone(server.START_REQUEST.status()['request_id'])
        finally:
            http.shutdown();http.server_close();worker.join()
            with server.LOCK:server.STATE.clear();server.STATE.update(state);server.START_REQUEST=old_request

if __name__=='__main__':unittest.main()
