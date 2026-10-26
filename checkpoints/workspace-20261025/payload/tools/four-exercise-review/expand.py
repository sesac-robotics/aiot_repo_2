from pathlib import Path
import json,sys,hashlib,numpy as np,cv2
sys.dont_write_bytecode=True
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/four-exercise-review-train-v1-20261025';sys.path.insert(0,str(B/'tools/error-enrichment'));from cache_sequences import Resolver;from split_guard import assert_training_allowed
old=json.loads((O/'review-plan.frozen.json').read_text());rows=json.loads((B/'output/squat-fixed-reference-combined-v3-20261024/score-inputs.frozen.json').read_text());known={x['id'] for x in old};suspect={old[n]['source_sha256'] for n in [2,19,23,42,45,52,53,59,60]};source={x['source_sha256']:x for x in old};res=Resolver();out=[]
for r in rows:
 if r['split']!='train' or r['exercise']=='squats' or r['id'] in known or not set(r['sources'])&suspect:continue
 seq,p=res.resolve(r)[0];m=json.loads(p.with_suffix('.json').read_text());z=np.load(p);sha=m['config']['source_sha256'];src=source[sha];fs=z['frame_numbers'];times=fs/float(m['config']['segment']['fps']);mask=np.ones(len(fs),bool);start=float(times[0]);end=float(times[-1])
 if r['family']=='R2D2':
  v=res.team[r['id'][5:]];start=v['start'];end=v['end'];mask=(times>=start)&(times<end)
 inds=np.flatnonzero(mask);assert len(inds)>=8
 d={k:r[k] for k in ['id','exercise','family','person','group','split','sources']};d.update(source_sha256=sha,source_path=src['source_path'],fps=src['fps'],width=src['width'],height=src['height'],window_seconds=[start,end],cache_path=str(p),cache_fingerprint=m['fingerprint'],cache_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),actual_native_frames=fs[mask].tolist(),cache_indices=inds.tolist(),previous_levels=r['levels'],previous_items=[x['item'] for x in r['items']],previous_banks=[x['bank'] for x in r['items']],selection='all_remaining_Train_intervals_of_sources_with_confirmed_or_suspected_label_or_tracking_issue; no_blanket_grade_propagation')
 out.append(d)
assert_training_allowed(out);p=O/'expanded-review-plan.frozen.json';assert not p.exists();p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(len(out))
s=(B/'tools/four-exercise-review/render.py').read_text().replace("review-plan.frozen.json","expanded-review-plan.frozen.json").replace("out=O/'review'","out=O/'expanded-review'").replace('range(32)','range((len(rows)+1)//2)').replace("n=pair*2+block","n=64+pair*2+block").replace("index=pair*2+block","index=64+pair*2+block").replace("page=f'review/", "page=f'expanded-review/").replace("render-receipts.json","expanded-render-receipts.json")
(B/'tools/four-exercise-review/render_expanded.py').write_text(s)
