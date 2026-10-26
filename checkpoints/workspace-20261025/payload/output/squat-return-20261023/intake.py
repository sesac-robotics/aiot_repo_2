from pathlib import Path
import urllib.request,json,hashlib,concurrent.futures
O=Path(__file__).resolve().parent;D=O/'cfrep';D.mkdir(exist_ok=True);tree=json.loads((O/'cfrep-tree.json').read_text());base='https://raw.githubusercontent.com/lc-leonardo/CFRep/main/'
paths=['README.md','CFRep/video_config.csv'];people=['m1','m3','m4','w1','w2']
for p in people:
 rows=[r for r in tree['tree'] if r['path'].startswith('CFRep/squat_front_'+p+'_') and r['path'].endswith('.mp4')];paths+=[r['path'] for r in rows]
 for r in rows:
  name=Path(r['path']).stem;paths+=[x['path'] for x in tree['tree'] if '/'+name+'/' in x['path'] and x['path'].endswith('_hrnet-w32_8xb64-210e_aic-256x192.json')]
def fetch(path):
 dest=D/Path(path).name
 if not dest.exists():urllib.request.urlretrieve(base+path,dest)
 return dict(path=str(dest),url=base+path,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),bytes=dest.stat().st_size)
with concurrent.futures.ThreadPoolExecutor(4) as pool:rows=list(pool.map(fetch,paths))
(O/'source-receipts.json').write_text(json.dumps(dict(repo='https://github.com/lc-leonardo/CFRep',commit=tree['sha'],records=rows,train_people=['m1','m3','m4','w1'],validation_people=['w2'],unopened_people=['m2','w4'],provider_labels_are_binary_not_return_labels=True),indent=2));print('downloaded',len(rows),flush=True)
