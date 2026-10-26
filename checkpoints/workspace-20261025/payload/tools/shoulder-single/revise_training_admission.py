"""Add an audited applicability overlay. Never mutate historical labels or splits."""
from pathlib import Path
import json,collections
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-single-audit-v1-20261025';audit={r['id']:r for r in json.loads((O/'train-provenance-audit.json').read_text())['rows']};original=json.loads((B/'output/five-exercise-recovery-20261022/held_round2/candidate_input_manifest.json').read_text());current={r['id']:r for r in json.loads((B/'output/shoulder-work-20261024/full-review-v1/combined-selected-labels.frozen.json').read_text())};out=[]
visual_bad=json.loads((O/'tracking-labels.frozen.json').read_text())
for row in original:
 if row['exercise']!='shoulder' or row['split']!='train':continue
 a=audit[row['id']]
 for index in [1,3]:
  note=row['items'][index]
  if not note['eligible'] or note['bank']!='front' or row['levels'][index] is None:continue
  hold=[];details=[]
  if a['status']!='exact_feature_provenance':hold.append(a['status'])
  if a.get('cache'):
   label=current.get(a['source_id']);arange=a['time_range'];interval=[max(arange[0],arange[1]-a['items'][0]['tail_feature_extent_seconds']+.1),arange[1]] if index==1 else arange
   # Historical labels reviewed a source or selected event, not every later cache interval.
   # A matched reviewed interval is only a review candidate, never automatic fit approval.
   windows=label.get('observed_events',[]) if index==1 and label else label.get('reviewed_windows',[]) if label else []
   matched=[w for w in windows if interval[0]>=w[0]-.07 and interval[1]<=w[1]+.07]
   if not matched:hold.append('precise_feature_window_not_covered_by_latest_visual_label')
   relevant=[f for f in a['quality']['flags'] if interval[0]-.05<=f['tick']/10<=interval[1]+.05]
   if relevant:hold.append('tracking_or_view_quality_uncertain')
   if a['person'] in ['shoulder_005','shoulder_008','shoulder_009']:hold.append('actual_rear_view_not_reviewed_front_model_bank')
   if a['source_id']=='subject_006_shoulder_bad_front' and row['id']==a['source_id'] and index==1:hold.append('direct_visual_tail_still_raised_not_a_return_event');details.append('base-train-02.jpg: native177/178/180; label2 cannot describe this tail; no movement-error grade inferred')
   actual_bad=[v for v in visual_bad if v['tracking_error'] and v['cache_sha256']==a['cache_sha256'] and interval[0]-.05<=v['native_frame']/a['source_fps']<=interval[1]+.05]
   if actual_bad:hold.append('direct_visual_tracking_misconnection')
   if a['source_id']=='subject_001_shoulder_good_front' and row['id']==a['source_id'] and index==3:hold.append('pickup_or_preparation_in_body_window')
  else:interval=None;matched=[]
  out.append(dict(id=row['id']+'_'+note['item'],source_row_id=row['id'],item=note['item'],person=row['person'],sources=row['sources'],split='train',original_grade=row['levels'][index],audited_training_grade=None if hold else row['levels'][index],status='quarantined_for_new_fit' if hold else 'window_matched_requires_final_tracking_review',holds=sorted(set(hold)),evidence=details,feature_window_seconds=interval,matched_visual_windows=matched,new_error_grade_inferred=False,model_fit_approved=False))
report={'schema':'shoulder-training-applicability-overlay-v2','rows':out,'counts':dict(collections.Counter(r['status'] for r in out)),'hold_counts':dict(collections.Counter(h for r in out for h in r['holds'])),'historical_artifacts_modified':False,'no_grade_zero_or_one_created':True,'scope':'Current conservative admission overlay; quarantine does not prove historical movement label was wrong.'}
target=O/'training-admission-overlay-v2.frozen.json'
if target.exists():raise ValueError('Frozen overlay already exists; create a new version')
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=False))
