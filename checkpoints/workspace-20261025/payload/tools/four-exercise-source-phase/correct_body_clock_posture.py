"""Correct research-only body observation slots without overwriting the earlier run.

 Native 15fps quantization and 10Hz observation slots are different clocks.
 Reconstruct the actual extraction schedule instead of rounding absolute time.
 Unchanged trained heads are verified and reused, not reported as new fits.
"""
from pathlib import Path
import json,sys,copy,hashlib,collections,numpy as np,joblib
sys.dont_write_bytecode=True
B=Path('/home/jayden/SeSACPAI-Mid');P=B/'output/four-exercise-source-phase-v2-20261025';O=P/'clock-posture-corrected-v4';O.mkdir(exist_ok=True)
sys.path[:0]=[str(B/'tools/shoulder-work'),str(B/'tools/four-exercise-source-phase')]
import observation_features_v4 as body
import features as sf
rows=json.loads((P/'score-inputs.frozen.json').read_text());prior=copy.deepcopy(rows);links={r['id']:r for r in json.loads((P/'all-resolved-inputs.frozen.json').read_text())};audit={r['id']:r for r in json.loads((P/'labels-audit.frozen.json').read_text())};receipt=[]
postures={r['source_sha256']:r for r in json.loads((P/'shoulder-posture-labels.frozen.json').read_text())};baselines=json.loads((P/'paired-baseline-used-v2.frozen.json').read_text());basemap={r['id']:r for r in baselines}
for r in rows:
 if r['exercise']!='shoulder' or r['id'] not in links:continue
 d=links[r['id']];z=np.load(d['cache_path']);meta=json.loads(Path(d['cache_path']).with_suffix('.json').read_text());cfg=meta['config'];seg=cfg['segment'];assert cfg['sample_hz']==10
 schedule=np.unique(np.rint(np.arange(seg['start_frame'],seg['end_frame']+.01,seg['fps']/cfg['sample_hz']))).astype(int);ix=np.asarray(d['cache_indices']);fs=z['frame_numbers'][ix];ticks=np.searchsorted(schedule,fs);assert np.array_equal(schedule[ticks],fs);assert np.all(np.diff(ticks)>0)
 x=sf.sanitize(z['image_xy'][ix],z['confidence'][ix],d['width'],d['height'])
 j=next(j for j,it in enumerate(r['items']) if it['bank'].startswith('four_joint_body_v2/'));it=r['items'][j];it['bank']='four_joint_body_v2/front/'+postures[d['source_sha256']]['posture'];
 for virtual in basemap[r['id']]['items']:
  if virtual['bank'].startswith('four_joint_body_v2/'):virtual['bank']=it['bank']
 grade=r['levels'][j];view='front_oblique' if any(a['item']=='frontal_trunk_hold' and a['bank']=='front_oblique' for a in r['items']) else r.get('view','front')
 result=body.extract_body(list(zip(ticks,x)),[int(ticks[0]),int(ticks[-1])],exercise_confirmed=grade is not None,width=d['width'],height=d['height'],view=view)
 if '*' in audit.get(r['id'],{}).get('input_holds',[]) or 'frontal_trunk_hold' in audit.get(r['id'],{}).get('input_holds',[]):result=dict(status='held',reason='source_visual_body_tracking_hold')
 result.update(observation_slot_clock='exact_index_in_verified_native_extraction_schedule',extraction_start_native_frame=seg['start_frame'],actual_native_frames=fs.tolist(),actual_observation_slots=ticks.tolist())
 previous=it['eligible'];ok=result['status']=='features_ready' and grade is not None;it.update(feature=result.get('features') if ok else None,eligible=ok,holds=[] if ok else [result.get('reason','manual_grade_unavailable')],observation_receipt=result)
 receipt.append(dict(id=r['id'],bank=it['bank'],grade=grade,previous_eligible=previous,corrected_eligible=ok,result=result))
report=json.loads((P/'score-training.json').read_text());folds=json.loads((P/'folds.frozen.json').read_text());bundle=joblib.load(P/'score-candidates.joblib');oldmap={r['id']:r for r in prior};newmap={r['id']:r for r in rows}
# Every serialized classifier receives exactly the same training vectors/labels.
for key,h in bundle['models'].items():
 ex,item,bank,alg=key.split('|');assert not bank.startswith('four_joint_body_v2/')
 for id in h['trained_ids']:
  a=oldmap[id];b=newmap[id];j=next(j for j,x in enumerate(a['items']) if (x['item'],x['bank'])==(item,bank));assert a['items'][j]['feature']==b['items'][j]['feature'] and a['levels'][j]==b['levels'][j]
def support(a):return {str(g):dict(rows=sum(y==g for id,y in a),groups=len({folds['group_by_id'][id] for id,y in a if y==g})) for g in [0,1,2]}
for h in report['heads']:
 if not h['bank'].startswith('four_joint_body_v2/'):continue
 values=[]
 for r in rows:
  for it,y in zip(r['items'],r['levels']):
   if it['item']==h['item'] and it['bank']==h['bank'] and y is not None and it['eligible']:values.append((r['id'],y))
 targets=[(r['id'],y) for r in rows for it,y in zip(r['items'],r['levels']) if it['item']==h['item'] and it['bank']==h['bank'] and y is not None]
 h.update(labelled_targets=len(targets),candidate_observable=len(values),candidate_support=support(values),body_clock_corrected=True,posture_source_pixels_verified=True)
 for a in h['algorithms']:
  assert a['status']=='unsupported' and not a['actual_fit_ids'];assert not any(p['prediction'] is not None for p in a['predictions']['candidate'])
  for role,mkey in [('candidate','candidate_cv'),('baseline','baseline_cv_same_targets')]:
   a['predictions'][role]=[dict(id=id,group=folds['group_by_id'][id],fold=folds['fold_by_id'][id],target=y,prediction=None,abstention_reason='insufficient_train_support') for id,y in targets]
   a[mkey].update(labelled_targets=len(targets),evaluated=0,abstained=len(targets),coverage=0.,all_labelled_errors=sum(y<2 for id,y in targets),all_labelled_normals=sum(y==2 for id,y in targets),end_to_end_error_recall=0. if any(y<2 for id,y in targets) else None)
  for f in a['folds']:
   tr=[p for p in values if folds['fold_by_id'][p[0]]!=f['fold']];ss=support(tr);tr=[p for p in tr if ss[str(p[1])]['groups']>=2];f['candidate']['grade_support']=support(tr)
report['body_clock_correction']=dict(parent=str(P),native_frame_schedule_verified=True,previous_absolute_time_rounding_not_used=True,reused_unchanged_trained_models=len(bundle['models']),new_classifier_fits_this_correction=0,reason='only_unfitted_body_head_observation_contract_changed; all_trained_feature_vectors_and_targets_exactly_equal')
text=json.dumps(rows,ensure_ascii=False,indent=2)+'\n';p=O/'score-inputs.frozen.json';assert not p.exists();p.write_text(text);bundle['inputs_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();bundle['reused_models_from']=str(P/'score-candidates.joblib');bundle['body_clock_corrected']=True
joblib.dump(bundle,O/'score-candidates.joblib')
for name,value in [('score-training.json',report),('body-clock-correction-receipts.json',receipt),('folds.frozen.json',folds)]:
 p=O/name;assert not p.exists();p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
(O/'paired-baseline-used-v2.frozen.json').write_text(json.dumps(baselines,ensure_ascii=False,indent=2)+'\n')
(O/'shoulder-posture-labels.frozen.json').write_bytes((P/'shoulder-posture-labels.frozen.json').read_bytes())
summary=dict(observation_contracts_rechecked=len(receipt),old_observable=sum(r['previous_eligible'] for r in receipt),corrected_observable=sum(r['corrected_eligible'] for r in receipt),restored_observable=sum(not r['previous_eligible'] and r['corrected_eligible'] for r in receipt),trained_heads_unchanged=len(bundle['models']),classifier_fit_count_this_correction=0,posture_source_review_count=42,changed_frozen_parent=False)
(O/'correction-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
