"""Frozen Train inputs; independent Test features prepared only after model freeze."""
import json,sys,copy,collections
from pathlib import Path
from fractions import Fraction
import numpy as np,joblib
B=Path('/home/jayden/SeSACPAI-Mid');BASE=B/'output/label-verified-five-expanded-20261024';D=B/'output/pushup-root-cause-20261024';O=B/'output/pushup-repaired-five-v2-20261024';O.mkdir(exist_ok=True)
sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'five-exercise-comparison')]
from common import sha
from semantic_features import describe as semantic
import three_items
from aihub_transfer import describe_common,JOINTS
enc=joblib.load(B/"output/error-enrichment-20261024/aihub-transfer-encoders.joblib")["models"]
rows=json.loads((BASE/('test-inputs.frozen.json' if '--test' in sys.argv else 'score-inputs.frozen.json')).read_text());plans={r['id']:r for r in json.loads((D/'crop-plan-v2.frozen.json').read_text())};team={r['id']:r for r in json.loads((B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json').read_text())['labels']};inventory={r['id']:r for r in json.loads((B/'output/r2d2-review-20261024/inventory.json').read_text())};audit=[]
for r in rows:
 if r['exercise']!='pushups':continue
 name=team[r['id'][5:]]['source_id'] if r['family']=='R2D2' else r['id']
 if name not in plans:continue
 plan=plans[name];p=D/'observations'/name/'primary.npz'
 if not p.exists():raise ValueError('Missing extraction '+name)
 meta=json.loads(p.with_suffix('.json').read_text());assert meta['config']['crop']==plan['crop'];assert meta['config']['source_sha256'] in r['sources'];z=np.load(p);assert str(z['fingerprint'])==meta['fingerprint'];seq=z['image_xy'];confidence=z['confidence']
 oldp=Path(plan['previous_cache']);oldmeta=json.loads(oldp.with_suffix('.json').read_text());oldz=np.load(oldp)
 assert oldmeta['config']['source_sha256'] in r['sources'] and str(oldz['fingerprint'])==oldmeta['fingerprint']
 assert np.array_equal(z['frame_numbers'],oldz['frame_numbers'])
 oldseq=oldz['image_xy'];oldconfidence=oldz['confidence']
 if r['family']=='R2D2':
  d=team[r['id'][5:]];times=z['frame_numbers']/float(Fraction(inventory[name]['fps']));mask=(times>=d['start'])&(times<d['end']);seq=seq[mask];confidence=confidence[mask];oldtimes=oldz['frame_numbers']/float(Fraction(inventory[name]['fps']));oldmask=(oldtimes>=d['start'])&(oldtimes<d['end']);oldseq=oldseq[oldmask];oldconfidence=oldconfidence[oldmask]
 # Label targets, source assignments and participant groups remain untouched.
 grades={i['item']:g for i,g in zip(r['items'],r['levels'])};manual_holds={i['item'] for i in r['items'] if any('direct_review_item_unobservable' in h for h in i.get('holds',[]))};original_items=copy.deepcopy(r['items']);notes=semantic(seq,confidence)
 oldnotes=semantic(oldseq,oldconfidence)
 for n,base in zip(notes,oldnotes):
  n['observation_source']='roi'
  if not n['eligible'] and base['eligible']:
   n.update(base,observation_source='original_visibility_fallback')
 legacy=three_items.describe('pushups',seq,None)
 oldlegacy=three_items.describe('pushups',oldseq,None)
 for n,base in zip(legacy,oldlegacy):
  n['observation_source']='roi'
  if not n['eligible'] and base['eligible']:n.update(base,observation_source='original_visibility_fallback')
 for n in legacy:
  n['bank']='side_legacy_roi_v1';n.pop('measurement',None);n.pop('higher_is_better',None)
 natives={'roi':describe_common(seq[:,JOINTS]),'original_visibility_fallback':describe_common(oldseq[:,JOINTS])}
 for n in list(notes):
  native=natives[n['observation_source']]
  if native is not None:
   task={'trunk_line':'pushup_trunk','descent_range':'pushup_descent'}.get(n['item'])
   if task and n['eligible']:
    m=enc[task];prob=float(m.predict_proba([native])[0,list(m.classes_).index(1)]);notes.append(dict(n,bank=n['bank']+'_aihub_transfer_v1',feature=n['feature']+[prob],native_condition_encoder=task,pretraining='AIHub_Training_only',provider_label_not_used_as_project_target=True))
 # Known old pose/feature pipeline remains available only in the prior frozen run.
 kept=[(i,g) for i,g in zip(r['items'],r['levels']) if not i['bank'].startswith('side')]
 for n in notes+legacy:
  g=grades[n['item']]
  if g is None or n['item'] in manual_holds:n.update(eligible=False,feature=None,holds=['manual_label_uncertain'])
  if not n['eligible'] or n['feature'] is None or not np.isfinite(n['feature']).all():
   n.update(eligible=False,feature=None)
  kept.append((n,g))
 r['items']=[i for i,g in kept];r['levels']=[g for i,g in kept];r['feature_revision']='pushup_roi_semantic_v1'
 audit.append(dict(id=r['id'],cache=str(p),cache_sha256=sha(p),original_cache=str(oldp),original_cache_sha256=sha(oldp),source_sha256=meta['config']['source_sha256'],crop=plan['crop'],n=len(seq),observation_sources={n['item']+':'+n['bank']:n.get('observation_source') for n in notes+legacy},targets_unchanged=all(g==grades[i['item']] for i,g in zip(r['items'],r['levels']))))
assert all(r['targets_unchanged'] for r in audit)
name='test-inputs.frozen.json' if '--test' in sys.argv else 'score-inputs.frozen.json';p=O/name
assert not p.exists() or json.loads(p.read_text())==rows;p.write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False));(O/('test-feature-provenance.json' if '--test' in sys.argv else 'feature-provenance.json')).write_text(json.dumps(audit,indent=2));print('PREPARED',name,len(rows),'repaired',len(audit),flush=True)
