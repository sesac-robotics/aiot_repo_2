from pathlib import Path
from functools import partial
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
s=ThreadingHTTPServer(('127.0.0.1',8797),partial(SimpleHTTPRequestHandler,directory=str(Path(__file__).resolve().parent)));print('http://127.0.0.1:8797/',flush=True);s.serve_forever()
