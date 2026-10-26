from pathlib import Path
import json,hashlib,datetime,os
import numpy as np
def json_scalar(value):
 if isinstance(value,np.generic):return value.item()
 if isinstance(value,np.ndarray):return value.tolist()
 raise TypeError(type(value).__name__)
B=Path('/home/jayden/SeSACPAI-Mid');O=Path(os.environ.get('POSTURE_RUN_OUTPUT',str(B/'output/error-enrichment-20261024')));O.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def save(name,obj):
 p=O/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=json_scalar));t.replace(p)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def status(stage,**kw):
 v=dict(stage=stage,time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),**kw);save('status.json',v);print(json.dumps(v,ensure_ascii=False),flush=True)
