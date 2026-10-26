"""Exercise the real end HTTP endpoint with a session longer than the old 60-second buffer."""
import unittest,tempfile,threading,json,urllib.request
from pathlib import Path
from unittest.mock import patch
import numpy as np
import server
class ExportTest(unittest.TestCase):
 def test_end_exports_ninety_seconds_and_result_rows(self):
  backup=dict(server.STATE)
  with tempfile.TemporaryDirectory() as folder:
   http=server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
   thread=threading.Thread(target=http.serve_forever,daemon=True);thread.start()
   try:
    z=np.load(Path(server.__file__).parent/'recordings/20261023-110200-964840/pose.npz')
    scored=server.score_samples(list(enumerate(z['image_xy'][35:86])),640,480,'squats','front',True)
    row={'number':1,'status':'test-result','coaching':{'messages':['saved cue']},'front_candidate':scored['front_candidate'],'method_comparison':scored['method_comparison'],'score_100':scored['score_100']}
    with server.LOCK:
     server.STATE.update(collecting=True,samples=[(n,np.full((33,2),np.nan)) for n in range(900)],width=640,height=480,repetitions=[row],movement={'count':1})
    with patch.object(server,'__file__',str(Path(folder)/'server.py')):
     request=urllib.request.Request(f'http://127.0.0.1:{http.server_port}/end',data=b'{}',headers={'Content-Type':'application/json'},method='POST')
     with urllib.request.urlopen(request) as response:self.assertEqual(response.status,200)
    saved=Path(server.STATE['saved_session_folder']);pose=np.load(saved/'pose.npz');state=json.loads((saved/'state.json').read_text())
    self.assertEqual(len(pose['frame_numbers']),900);self.assertEqual(int(pose['frame_numbers'][0]),0)
    self.assertEqual(state['repetitions'],[row])
    comparison=state['repetitions'][0]['method_comparison']
    self.assertEqual(len(comparison['item_comparisons']),4)
    self.assertIn('knee_inward',comparison['feature_evidence'])
    self.assertEqual(state['repetitions'][0]['score_100']['ai']['maximum'],100)
    self.assertIsNone(state['repetitions'][0]['score_100']['ai']['total'])
   finally:
    http.shutdown();http.server_close();thread.join()
    with server.LOCK:server.STATE.clear();server.STATE.update(backup)
