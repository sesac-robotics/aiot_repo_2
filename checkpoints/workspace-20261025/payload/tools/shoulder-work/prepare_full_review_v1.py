from pathlib import Path
import json,hashlib,collections,re
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-work-20261024/full-review-v1';O.mkdir(exist_ok=True)
pool=json.loads((B/'output/overnight-20261020/score_candidates.json').read_text());pool={r['id']:r for r in pool if r['exercise']=='shoulder'}
root=B/'data/A Multi-View Raw Video Dataset of Seven Fitness Ex/Dataset Exercise-wise/shoulder';rows=[]
for p in sorted(root.rglob('*.mp4')):
 r=pool[p.stem];num=int(re.search(r'subject_(\d+)',p.stem)[1]);rows.append(dict(id=p.stem,path=str(p),source_sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),person=r['person'],recording_group=r['recording_group'],view=p.parent.name,provider_tag=p.parent.parent.name,split=r['split'],owner='model_evaluation' if num<=12 else 'shoulder',review_state='pending',lower_return={'grade':None,'status':'unreviewed'},frontal_trunk_hold={'grade':None,'status':'unreviewed'}))
assert len(rows)==150
manifest=dict(schema='shoulder-full150-inventory-v1',sources=rows,summary={'total':len(rows),'roles':dict(collections.Counter(r['split'] for r in rows)),'owners':dict(collections.Counter(r['owner'] for r in rows))},policy='Existing90Train/30validation/30historically-designatedTest unchanged; no source reassignment. Provider good/bad is not either item grade. Side lumbar alignment cannot supervise frontal trunk. Review all150,fit onlyTrain.',previous_candidate_sources=[str(B/'output/overnight-20261020/score_candidates.json')])
(O/'inventory.frozen.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(manifest['summary'])
