"""Freeze directly reviewed protocol labels, then extract isolated source windows."""
import datetime,json,sys
from pathlib import Path
B=Path(__file__).resolve().parents[2];sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'five-exercise-comparison')]
from common import sha
from split_guard import POLICY,assert_training_allowed
import extract
R=B/'output/shoulder-common-support-audit-20261024/full-review-v1'
O=R/'protocol-model-v1'
def save(n,x):
 p=O/n;assert not p.exists();p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False))
def main():
 O.mkdir(exist_ok=True)
 invpath=B/'output/shoulder-work-20261024/full-review-v1/inventory.frozen.json';inv={r['id']:r for r in json.loads(invpath.read_text())['sources']}
 partnerpath=B/'output/shoulder-work-20261024/full-review-v1/dense-visual-labels.json'
 partners=json.loads(partnerpath.read_text());labels=[]
 for r in partners:
  if r['view']!='side' or r['movement_protocol'] not in ['shoulder_press','non_press_arm_extension']:continue
  # Only a directly reviewed contiguous window; never bridge unseen gaps.
  a,b=max(r['reviewed_windows'],key=lambda x:x[1]-x[0]);src=inv[r['id']]
  labels.append(dict(src,protocol_label=r['movement_protocol'],start=a,end=b,
     evidence=r['reviewed_pages'],review_origin='shoulder_thread_dense_visual',label_source_sha256=sha(partnerpath)))
 own=[('005','good','non_press_arm_extension',0,4),('010','bad','non_press_arm_extension',0,4),
      ('005','bad','shoulder_press',2,4),('006','bad','shoulder_press',0,4),
      ('010','good','shoulder_press',2,4),('001','bad','shoulder_press',3.6,7.4),
      ('001','good','shoulder_press',31.6,33.4),('006','good','shoulder_press',32.6,34.4)]
 for n,q,label,a,b in own:
  rid=f'subject_{n}_shoulder_{q}_side';src=inv[rid];sheet=R/(rid+'-dense-01.jpg');assert sheet.exists()
  labels.append(dict(src,protocol_label=label,start=a,end=b,evidence=[str(sheet)],
      review_origin='model_thread_dense_visual',label_source_sha256=sha(sheet)))
 for r in labels:
  r['evaluation_group']='conservative_subject_005_006' if r['person'] in ['shoulder_005','shoulder_006'] else r['person']
  r['scope']='observed_selected_window_only_not_provider_or_whole_video_grade'
  r['severity_grade']=None;r['production_replacement']=False;r['sources']=[r['source_sha256']]
  assert r['split']==inv[r['id']]['split']
 assert_training_allowed([r for r in labels if r['split']=='train'])
 save('labels.frozen.json',dict(protocol='side_press_vs_nonpress_arm_extension_v1',labels=labels,
      partner_labels_snapshot_sha256=sha(partnerpath),new_severity_grades=0,
      no_person_identity_inference=True,conservative_group_reason='Source005 side clothing/recording continuity differs from005front and resembles006; identity unverified, merge for CV without changing source roles.'))
 save('plan.frozen.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
      criterion='observed_side_arm_movement_protocol_only',classes=['shoulder_press','non_press_arm_extension'],
      not_a_return_or_trunk_severity_model=True,source_roles_preserved=True,policy_sha256=sha(POLICY),
      features='highest_observation_coverage_single_arm_chain; elevation/elbow/wrist-up q10/q50/q90/span; no visibility, provider labels, duration, identity or trunk feature',
      observation='actual10Hz source frames, required single chain>=80percent and>=10observed, image2percent border, no missing interpolation',
      estimator='Train-only StandardScaler+LogisticRegression C1 max_iter2000 fixed probabilitycutoff.5',
      sample_weights='equal group influence then balance observed protocol classes; normalize average1',
      validation='LeaveOneGroupOut Train; historical development/Test only after frozen candidate; final Test remains unopened',
      rule='single upperarm elevation q10 cutoff, Train-only equal-group balanced error minimization; per-fold calibration excludes held group',
      no_numeric_selection_on_development_or_Test=True,production_changes=False))
 receipts=[]
 for r in labels:
  fps=15.0
  row=dict(id=r['id'],video_sha256=r['source_sha256'],required_views=['side'],
      segment=dict(start_frame=round(r['start']*fps),end_frame=round(r['end']*fps),fps=fps))
  rec=extract.extract_view(row,'side',r['path'],O/'observations')
  receipts.append(dict(id=r['id'],extraction=rec));(O/'extraction-progress.json').write_text(json.dumps(receipts,indent=2));print('PROTOCOL_POSE',r['id'],r['split'],rec['frames'],flush=True)
 save('extraction.frozen.json',receipts);print('DONE',len(receipts),flush=True)
if __name__=='__main__':main()
