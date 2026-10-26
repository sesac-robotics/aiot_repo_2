import json,datetime,sys,copy,joblib
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');OLD=B/'output/pushup-repaired-five-v2-20261024';O=B/'output/paired-enriched-five-v2-20261024';SUB=O/'return-only-fit';sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import assert_training_allowed,POLICY
read=lambda p:json.loads(p.read_text());rows=read(O/'score-inputs.frozen.json');oldrows=read(OLD/'score-inputs.frozen.json');assert rows[:len(oldrows)]==oldrows;assert sha(O/'test-inputs.frozen.json')==sha(OLD/'test-inputs.frozen.json')
a=joblib.load(OLD/'score-candidates.joblib');b=joblib.load(SUB/'score-candidates.joblib');tr=read(OLD/'score-training.json');newtr=read(SUB/'score-training.json');assert newtr['completed'];key=('pushups','return_extension','front');assert set(b['heads'])=={key};before={k:joblib.hash(h) for k,h in a['heads'].items()};a['heads'][key]=b['heads'][key];a['labels_sha256']=sha(O/'score-inputs.frozen.json');a['parent_run']=str(OLD);a['changed_head_only']=key
assert all(joblib.hash(h)==before[k] for k,h in a['heads'].items() if k!=key)
for i,h in enumerate(tr['heads']):
 if tuple(h['key'])==key:tr['heads'][i]=newtr['heads'][0]
tr.update(trained_heads=len(a['heads']),gate_passed=sum(h.get('candidate_gate_passed',False) for h in tr['heads']),only_refitted_head=list(key),unchanged_heads_copied=len(a['heads'])-1,all_source_families_retained=True)
ids={i for h in a['heads'].values() for i in h['trained_ids']};byid={r['id']:r for r in rows};assert_training_allowed([byid[i] for i in ids]);test=read(O/'test-inputs.frozen.json');assert not {s for r in rows for s in r['sources']}&{s for r in test for s in r['sources']}
p=O/'score-candidates.joblib';assert not p.exists();joblib.dump(a,p);(O/'score-training.json').write_text(json.dumps(tr,ensure_ascii=False,indent=2));(O/'model-freeze-before-test.json').write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),ai_model_sha256=sha(p),training_input_sha256=sha(O/'score-inputs.frozen.json'),test_input_sha256=sha(O/'test-inputs.frozen.json'),policy_sha256=sha(POLICY),unchanged_heads=len(a['heads'])-1,refitted_heads=1,source_overlap=0,live_service_replaced=False),indent=2));print('MERGED_FROZEN',len(a['heads']),flush=True)
