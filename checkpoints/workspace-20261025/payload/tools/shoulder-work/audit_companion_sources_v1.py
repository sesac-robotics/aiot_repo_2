from pathlib import Path
import json,hashlib,datetime,subprocess,os
from source_admission_v1 import assess_source
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-work-20261024/source-intake-v2-20261025';O.mkdir(exist_ok=True)
I=B/'output/shoulder-work-20261024/full-review-v1/inventory.frozen.json';sources={r['id']:r for r in json.loads(I.read_text())['sources']};groups={r['id']:r for r in json.loads((B/'output/shoulder-work-20261024/source-alias-audit-v1-20261025/source-groups.frozen.json').read_text())}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
companion=[]
for p in sorted((B/'data/A Multi-View Raw Video Dataset of Seven Fitness Ex/Dataset Subject-wise').glob('*/shoulder/*/*/*.mp4')):
 s=sources[p.stem];h=sha(p);assert h==s['source_sha256']
 companion.append(dict(path=str(p),source_id=s['id'],sha256=h,original_role=s['split'],additional_independent_source=False))
assert len(companion)==150
segments=[r for r in json.loads((B/'data/error-enrichment-20261024/annotations/segments.frozen.json').read_text()) if r['exercise']=='shoulder']
clips={r['id']:r for r in json.loads((B/'data/error-enrichment-20261024/clip-manifest.json').read_text())};derivatives=[]
for r in segments:
 c=clips[r['id']];assert sha(c['clip'])==c['sha256'];assert c['source_sha256']==sources[r['source_id']]['source_sha256']
 a=assess_source(r,sources,groups)
 derivatives.append(dict(id=r['id'],parent_source_id=r['source_id'],parent_sha256=r['source_sha256'],derivative_sha256=c['sha256'],source_admission=a,
    label_levels_preserved=r['levels'],label_origin=r['annotation_status'],
    new_label=False,used_for_fit=False,additional_independent_source=False))
# Preserve historical flags; validate canonical roles before any future fit.
assert sum(r['source_admission']['training_source_allowed'] for r in derivatives)==7
assert sum('canonical_validation_not_training' in r['source_admission']['holds'] for r in derivatives)==3
result=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),subjectwise_companion_files=150,all_companion_same_id_hash_matches=True,
    additional_unique_raw_sources=0,enrichment_derivative_clips=10,derivative_parent_sources=6,canonical_Train_clips=7,canonical_validation_clips=3,
    historical_review_flags_do_not_override_source_roles=True,model_fit=False,production_changed=False,camera_capture=False,Test_video_content_decoded=False,
    companion_sources=companion,enrichment_derivatives=derivatives)
p=O/'companion-and-derivative-audit.frozen.json';assert not p.exists();p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
p=subprocess.run([str(B/'.posture-score-verify/bin/python'),'-m','unittest','discover','-s','tools/shoulder-work','-p','test_source_admission_v1.py','-v'],cwd=B,env=env,capture_output=True,text=True);assert p.returncode==0;(O/'tests.log').write_text(p.stdout+p.stderr)
(O/'verification.json').write_text(json.dumps(dict(functional_tests_passed=7,canonical_validation_clips_rejected_for_training=3,new_grade_labels=0,model_fit=False,
    source_admission_sha256=sha(B/'tools/shoulder-work/source_admission_v1.py'),inventory_sha256=sha(I)),indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['companion_sources','enrichment_derivatives']},ensure_ascii=False))
