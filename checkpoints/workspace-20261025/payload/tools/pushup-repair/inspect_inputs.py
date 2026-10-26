import sys,json,collections
from pathlib import Path
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'tools/label-recheck')]
from cache_sequences import Resolver
rs=[r for r in json.loads((B/'output/label-verified-five-expanded-20261024/score-inputs.frozen.json').read_text()) if r['exercise']=='pushups'];pool={r['id']:r for r in json.loads((B/'output/overnight-20261020/score_candidates.json').read_text())};res=Resolver();out=[]
for r in rs:
 if r['family']=='R2D2':continue
 pairs=res.resolve(r)
 if not pairs:continue
 seq,p=pairs[0];d=json.loads(p.with_suffix('.json').read_text());lookup=pool.get(r['id']) or next((v for k,v in pool.items() if r['id'].startswith(k)),None)
 out.append(dict(id=r['id'],family=r['family'],person=r['person'],split=r['split'],levels=r['levels'][:4],cache=str(p),source=lookup['path'] if lookup else None,segment=d.get('config',{}).get('segment'),n=len(seq),source_sha256=r['sources']))
(B/'output/pushup-root-cause-20261024/train-input-inventory.json').write_text(json.dumps(out,indent=2));print(len(out));print(json.dumps(out[:3]+[r for r in out if 'direct_rep' in r['id']][:2],indent=2))
