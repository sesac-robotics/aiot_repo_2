"""Local HTTP integration smoke test with a real held-out pose sequence."""
from pathlib import Path
import json,threading,unittest
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import numpy as np
from api_server import make_server
BASE=Path(__file__).resolve().parent

class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=make_server(0);cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start();cls.url=f'http://127.0.0.1:{cls.server.server_port}'
    @classmethod
    def tearDownClass(cls):cls.server.shutdown();cls.server.server_close();cls.thread.join()
    def test_health_and_real_prediction(self):
        with urlopen(self.url+'/health') as response:self.assertEqual(json.load(response)['targets'],['shallow','back not straight'])
        vid=json.loads((BASE/'outputs/splits.json').read_text())['test']['video_ids'][0]
        cache=np.load(BASE/'cache'/f'{vid}.npz')
        payload=json.dumps(dict(landmarks=cache['landmarks'].tolist(),aspect=float(cache['aspect']))).encode()
        with urlopen(Request(self.url+'/predict',data=payload,headers={'Content-Type':'application/json'})) as response:
            result=json.load(response);self.assertEqual(result['status'],'ok');self.assertEqual(len(result['scores']),2)
    def test_bad_payload_rejected(self):
        for body in [[],{'landmarks':[]}]:
            with self.assertRaises(HTTPError) as error:urlopen(Request(self.url+'/predict',data=json.dumps(body).encode()))
            self.assertEqual(error.exception.code,400)

if __name__=='__main__':unittest.main(verbosity=2)
