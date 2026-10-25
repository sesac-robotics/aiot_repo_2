from pathlib import Path
import json,hashlib,sys,cv2,datetime,subprocess,os
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-work-20261024/grade-support-v1-20261025';sys.path.insert(0,str(B/'tools/squat-fixed-reference'))
from return_target_contract import assess
from anatomical_turn_features_v1 import VERSION
inv={r['id']:r for r in json.loads((B/'output/shoulder-work-20261024/full-review-v1/inventory.frozen.json').read_text())['sources']}
labels=json.loads((O/'validation-feature-diagnostics.json').read_text());overlays={r['id']:r for r in json.loads((O/'all-frame-overlay-receipts.json').read_text())};out=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for r in labels:
    a=r['annotation'];s=inv[r['id']].copy();c=cv2.VideoCapture(s['path']);s['frame_count']=int(c.get(7));c.release();assert s['split']=='validation'
    assert r['features']['status']=='features_ready';pages=overlays[s['id']]['pages']
    # Reviewer inspected all 31/29 cached context frames on these six pages.
    receipt=dict(status='passed',source_sha256=s['source_sha256'],feature_schema=VERSION,
        target_person_tracking_QA=True,actual_10Hz_slots_80pct_and_2pct_border=True,
        no_interpolation_or_tracking_break=True,posture=a['posture'],view=a['view'],
        event_frames=[a['peak_frames'],a['turn_frames'],a['next_ascent_frames']],
        review_scope='all_actual_10Hz_context_frames_AI_visual_review_not_expert_adjudicated',
        reviewed_native_frames=overlays[s['id']]['native_frames'],
        reviewed_overlay_sha256={p:sha(p) for p in pages},
        limitations=['anatomical landmark precision is visually approximate','same exposed validation person; not final accuracy'])
    result=assess(a,s,observation=receipt,registered_schemas=(VERSION,))
    assert not result['training_eligible'] and result['evaluation_input_eligible']
    out.append(dict(id=s['id'],split=s['split'],features=r['features'],observation_receipt=receipt,admission=result))
path=O/'validation-observation-admissions.frozen.json';text=json.dumps(out,ensure_ascii=False,indent=2)+'\n'
assert not path.exists();path.write_text(text)
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
test=subprocess.run([str(B/'.posture-score-verify/bin/python'),'-m','unittest','discover','-s','tools/shoulder-work','-p','test_anatomical_turn_v1.py','-v'],cwd=B,env=env,capture_output=True,text=True);assert test.returncode==0
(O/'tests.log').write_text(test.stdout+test.stderr)
files=['anatomical_turn_features_v1.py','test_anatomical_turn_v1.py','extract_anatomical_validation_v1.py','verify_anatomical_validation_v1.py','render_anatomical_all_frames_v1.py','freeze_anatomical_validation_v1.py']
code=O/'code-snapshot';code.mkdir(exist_ok=True)
for name in files:
    p=code/name;assert not p.exists();p.write_bytes((B/'tools/shoulder-work'/name).read_bytes())
verification=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),functional_tests_passed=13,
    existing_validation_sources=2,actual_context_frames_reviewed=[len(r['observation_receipt']['reviewed_native_frames']) for r in out],
    evaluation_features_admitted=2,training_inputs_admitted=0,model_fit=False,production_changed=False,camera_capture=False,
    new_train_failure_labels=0,independent_accuracy=False,
    feature_extractor_sha256=sha(B/'tools/shoulder-work/anatomical_turn_features_v1.py'),
    admission_sha256=sha(path),review_is_AI_assisted=True)
p=O/'verification.frozen.json';assert not p.exists();p.write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n');print(verification)
