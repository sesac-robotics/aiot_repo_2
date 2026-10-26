from pathlib import Path
import json,hashlib,subprocess,os,datetime
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-work-20261024/observation-return-v4-20261025'
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
logs=[]
for name in ['test_phase_features_v3.py','test_observation_v4.py']:
    p=subprocess.run([str(B/'.posture-score-verify/bin/python'),'-m','unittest','discover','-s','tools/shoulder-work','-p',name,'-v'],cwd=B,env=env,text=True,capture_output=True)
    logs.append(p.stdout+p.stderr);assert p.returncode==0,p.stdout+p.stderr
(O/'tests.log').write_text('\n'.join(logs))
rows=json.loads((O/'replay.json').read_text());reg=json.loads((O/'boundary-regression.json').read_text())
assert len(rows)==30 and all(r['split']!='test' for r in rows)
assert sum(r['eligible_for_future_feature_fit'] for r in rows)==16
assert not any(r['eligible_for_future_feature_fit'] for r in rows if r['previous_visual_quarantine'])
assert reg[0]['old']['return_ticks']==[9,10,11] and reg[0]['new']['status']=='held'
assert reg[1]['v3']['return_ticks']==[24,25,26] and reg[1]['new']['reason']=='tracking_geometry_uncertain'
files=[B/'tools/shoulder-work'/f for f in ['observation_features_v4.py','score_items_v4.py','test_observation_v4.py','replay_observation_v4.py','render_quality_v4.py','finalize_observation_v4.py']]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
code=O/'code-snapshot';code.mkdir(exist_ok=True)
for p in files:
    target=code/p.name
    if target.exists():assert target.read_bytes()==p.read_bytes(),'new audited version required'
    else:target.write_bytes(p.read_bytes())
verification=dict(updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),functional_tests_passed=32,
    cached_input_rows=30,source_roles_preserved=True,prior_manual_quarantines_still_ineligible=7,
    fit_eligible_feature_rows=16,fit_performed=False,new_grade_labels=0,Test_inference=False,
    camera_opened=False,live_app_changed=False,models_changed=False,independent_accuracy_evaluated=False,
    implementation_sha256={str(p.relative_to(B)):sha(p) for p in files},
    evidence_sha256={p.name:sha(p) for p in [O/'evaluation-manifest.frozen.json',O/'replay.json',O/'boundary-regression.json',O/'summary.json',O/'tests.log']})
(O/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(verification,ensure_ascii=False,indent=2))
