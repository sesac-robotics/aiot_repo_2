"""Live-session continuity and local HTTP checks; no camera required."""
import unittest,threading,json,urllib.request,urllib.error
import numpy as np
from live_app.engine import TemporalGate,CameraReader,LiveEngine
from live_app.server import make_server

class GateTests(unittest.TestCase):
 def setUp(self):self.p=np.ones((33,4),np.float32);self.g=TemporalGate(3)
 def test_missing_and_gap_clear(self):
  for i in range(3):self.g.append(self.p,i*.2,1,[11])
  self.assertEqual(len(self.g.points),3)
  self.assertEqual(self.g.append(None,.7,1,[11]),'pose_not_visible');self.assertEqual(len(self.g.points),0)
  self.g.append(self.p,1,1,[11]);self.g.append(self.p,3,1,[11]);self.assertEqual(len(self.g.points),1)
 def test_aspect_and_low_visibility(self):
  self.g.append(self.p,0,1,[11]);self.g.append(self.p,.2,2,[11]);self.assertEqual(len(self.g.points),1)
  p=self.p.copy();p[11,3]=0
  for i in range(3):self.assertEqual(self.g.append(p,.4+i*.2,2,[11]),'pose_not_visible')
  self.assertEqual(len(self.g.points),0)
 def test_display_requires_repeat_and_current(self):
  self.assertEqual(self.g.stabilize({'detected_errors':['x']}),[])
  self.assertEqual(self.g.stabilize({'detected_errors':['x']}),['x'])
  self.assertEqual(self.g.stabilize({'detected_errors':[]}),[])
  self.g.clear();self.assertEqual(self.g.stabilize({'detected_errors':['x']}),[])
 def test_stop_before_start_and_idle_reset(self):
  CameraReader('0').stop();engine=LiveEngine();engine.reset();self.assertEqual(engine.snapshot()['status'],'idle')

class HTTPTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.server=make_server(0);cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start();cls.base=f'http://127.0.0.1:{cls.server.server_port}'
 @classmethod
 def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
 def post(self,path,body,origin=None):
  headers={'Content-Type':'application/json'}
  if origin:headers['Origin']=origin
  return urllib.request.urlopen(urllib.request.Request(self.base+path,json.dumps(body).encode(),headers),timeout=5)
 def test_page_and_frame(self):
  with urllib.request.urlopen(self.base) as r:self.assertIn('실시간 자세 오류 테스트',r.read().decode())
  with urllib.request.urlopen(self.base+'/api/frame.jpg') as r:self.assertEqual(r.headers['Content-Type'],'image/jpeg');self.assertTrue(r.read().startswith(b'\xff\xd8'))
 def test_reject_bad_exercise_and_foreign_origin(self):
  for body,origin,code in [({'exercise':'invalid','source':'0'},None,400),({},'https://example.com',403),([],None,400)]:
   with self.assertRaises(urllib.error.HTTPError) as error:self.post('/api/start',body,origin)
   self.assertEqual(error.exception.code,code)
 def test_stop_clears_prediction(self):
  self.server.engine.publish(result={'detected_errors':['x']},last_result={'detected_errors':['x']},buffer_frames=32)
  with self.post('/api/stop',{}) as r:state=json.load(r)
  self.assertIsNone(state['result']);self.assertEqual(state['buffer_frames'],0);self.assertEqual(state['last_result']['detected_errors'],['x'])

if __name__=='__main__':unittest.main()
