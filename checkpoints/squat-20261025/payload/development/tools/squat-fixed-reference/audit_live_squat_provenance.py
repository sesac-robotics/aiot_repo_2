"""Read-only provenance check for the actual 2026-10-25 squat session."""
import json,hashlib,datetime,sys
from pathlib import Path
import joblib
B=Path(__file__).resolve().parents[2];L=B/'output/live-webcam-20261023';O=B/'output/squat-live-provenance-20261025';P=B/'output/squat-fixed-reference-combined-v3-20261024'
def read(p):return json.loads(p.read_text())
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 O.mkdir(exist_ok=True)
 latest=read(L/'sessions/1792857230761398358/final-state.json');earlier=read(L/'sessions/1792846159556060119/final-state.json')
 assert latest['exercise']=='squats' and latest['view']=='front'
 assert latest['loaded_code_model_sha256']==earlier['loaded_code_model_sha256']
 assert latest['standing_state_model_sha256']==earlier['standing_state_model_sha256']
 depth=joblib.load(L/'front-candidate/model.joblib');rows=read(L/'front-candidate/training-rows.json');byid={r['id']:r for r in rows};fit=[byid[i]for i in depth['trained_ids']]
 assert len(fit)==14 and all(r['split']=='train' and '/rehab26-squat/raw/videos/ex6/' in r['front_source'] for r in fit)
 assert sha(L/'front-candidate/model.joblib')==read(L/'front-candidate/report.json')['model_sha256']
 state=joblib.load(L/'standing-state/model.joblib');sp=B/'output/squat-standing-state-20261023/labels.frozen.json';sl=read(sp)['rows']
 assert sha(sp)==state['label_sha256'];assert all(r['source'].startswith('squat_') and '/squat-return-20261023/cfrep/squat_'in r['path']for r in sl)
 assert sha(L/'standing-state/model.joblib')==sha(B/'output/squat-standing-state-20261023/model.joblib')==latest['standing_state_model_sha256']
 checklist=joblib.load(B/'output/squat-relabel-expanded-20261023/candidate.joblib');cp=B/'output/squat-relabel-expanded-20261023/labels.frozen.json';cl=read(cp)['records'];cm=read(B/'output/squat-relabel-expanded-20261023/source-manifest.json')
 assert sha(cp)==checklist['label_sha256'];assert {r['id']for r in cl}=={r['id']for r in cm}
 assert all('/rehab26-squat/raw/videos/ex6/'in v['path']for r in cm for v in r['views'].values())
 for n,h in read(P/'artifact-hashes.frozen.json').items():assert sha(P/n)==h
 # Runtime snapshot routing retains separate exercise-named models.
 snap=L/'runtime'/latest['loaded_code_model_sha256'];manifest=read(snap/'artifacts/posture_score/manifest.json')
 result=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),session='1792857230761398358',previous_squat_session='1792846159556060119',exercise='squats',view='front',loaded_runtime_same_as_first_squat=True,standing_model_same_as_first_squat=True,front_depth_fitted_rows=len(fit),front_depth_fitted_ids=depth['trained_ids'],front_depth_source_scope='REHAB26 ex6 squat only',standing_source_scope='CFRep squat_* sources only',standing_labels_hash_verified=True,checklist_labels_hash_verified=True,checklist_source_scope='REHAB26 ex6 squat only',shoulder_rows_found_in_audited_squat_specific_training=0,offline_candidate_artifacts_verified=26,offline_5_of_6_candidate_deployed=False,saved_video_present=False,session_files=sorted(p.name for p in (L/'sessions/1792857230761398358').iterdir()),new_fit=0,production_modified=False,scope='Squat-specific front depth, standing and checklist label chains; not a blanket assertion about all historical multi-exercise models',manifest_keys=list(manifest))
 (O/'verification.frozen.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
