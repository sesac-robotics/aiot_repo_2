"""Resolve exact original observations, rejecting stale/source-mismatched caches."""
from pathlib import Path
import json,collections
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/error-enrichment-20261024'
class Resolver:
 def __init__(self):
  self.paths=collections.defaultdict(list)
  for p in (B/'output').glob('**/*.npz'):
   if p.name not in ['primary.npz','side.npz','front.npz'] or any(s in str(p) for s in ['/snapshots/','/before/','/team_install','/team_share','/code-snapshot/']):continue
   self.paths[p.parent.name].append(p)
  self.team={r['id']:r for r in json.loads((B/'data/r2d2-team-20261023/annotations/labels.frozen.json').read_text())['labels']};self.new={r['id']:r for n in ['segments.frozen.json','rehab-segments.frozen.json'] for r in json.loads((B/'data/error-enrichment-20261024/annotations'/n).read_text())};self.inventory={r['id']:r for r in json.loads((B/'output/r2d2-review-20261024/inventory.json').read_text())}
 def resolve(self,r):
  if r['family']=='R2D2':
   d=self.team[r['id'][5:]];p=B/'output/integrated-five-20261024/team-observations'/d['source_id']/'side.npz';z=np.load(p);from fractions import Fraction
   times=z['frame_numbers']/float(Fraction(self.inventory[d['source_id']]['fps']));return [(z['image_xy'][(times>=d['start'])&(times<d['end'])],p)]
  if r['id'] in self.new:
   d=self.new[r['id']];p=O/'observations'/d['review_id']/'primary.npz';z=np.load(p);times=z['frame_numbers']/d['fps'];return [(z['image_xy'][(times>=d['start'])&(times<d['end'])],p)]
  paths=self.paths.get(r['id'],[])
  if not paths and r['id'].endswith('_held3_score'):paths=self.paths.get(r['id'][:-12],[])
  out=[]
  for p in sorted(paths,key=lambda p:p.stat().st_mtime_ns,reverse=True):
   meta=p.with_suffix('.json')
   if not meta.exists():continue
   d=json.loads(meta.read_text());config=d.get('config',{})
   if config.get('source_sha256') not in r['sources']:continue
   z=np.load(p)
   if 'fingerprint' not in z.files or str(z['fingerprint'])!=d.get('fingerprint'):continue
   out.append((z['image_xy'],p))
  return out
