"""Audited versions of all existing Train inputs; frozen parents remain untouched."""
from pathlib import Path
import json,sys,copy,hashlib,collections,numpy as np
sys.dont_write_bytecode=True
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/four-exercise-source-phase-v2-20261025';P=B/'output/four-exercise-review-train-v1-20261025/reconciled-v2'
sys.path[:0]=[str(B/'tools/four-exercise-source-phase'),str(B/'tools/shoulder-work'),str(B/'tools/error-enrichment'),str(B/'five-exercise-comparison')]
import features as sf
from split_guard import assert_training_allowed
import observation_features_v4 as body
import held_geometry,geometry,plank_item_observation
parents=json.loads((P/'score-inputs.frozen.json').read_text());rows=copy.deepcopy(parents);old=copy.deepcopy(parents)
links={r['id']:r for r in json.loads((O/'all-resolved-inputs.frozen.json').read_text())};audit={r['id']:r for r in json.loads((O/'labels-audit.frozen.json').read_text())};events=[];receipts=[]
def add(r,item,bank,grade,result,baseline_item):
 f=result.get('feature',result.get('features'));ok=result.get('status')=='features_ready' and grade is not None
 x=dict(item=item,bank=bank,feature=f if ok else None,eligible=ok,holds=[] if ok else [result.get('reason','manual_grade_unavailable')],features_contain_no_rule_grades=True,observation_receipt=result)
 r['items'].append(x);r['levels'].append(grade)
 # Baseline keeps its original feature vector and eligibility. Validation must
 # feed this vector, not the revised vector, to the old feature classifier.
 b=copy.deepcopy(baseline_item) if baseline_item else dict(item=item,feature=None,eligible=False,holds=['no_equivalent_original_interval_feature'])
 b.update(bank=bank,paired_baseline_only=True);o=oldmap[r['id']];o['items'].append(b);o['levels'].append(grade)
 receipts.append(dict(id=r['id'],exercise=r['exercise'],item=item,bank=bank,grade=grade,eligible=ok,result=result))
oldmap={r['id']:r for r in old}
for r in rows:
 d=links[r['id']];v=audit.get(r['id'],{});z=np.load(d['cache_path']);assert hashlib.sha256(Path(d['cache_path']).read_bytes()).hexdigest()==d['cache_sha256'];ix=np.array(d['cache_indices']);frames=z['frame_numbers'][ix];q=z['confidence'][ix];x=sf.sanitize(z['image_xy'][ix],q,d['width'],d['height'])
 assert frames.tolist()==d['actual_native_frames'];r['source_observation_link']=str(O/'all-resolved-inputs.frozen.json');r['audit_version']='four-exercise-source-phase-v2'
 for j,it in enumerate(r['items']):
  previous=r['levels'][j]
  if it['item'] in v.get('grade_overrides',{}):r['levels'][j]=v['grade_overrides'][it['item']]
  held='*' in v.get('input_holds',[]) or it['item'] in v.get('input_holds',[])
  if held or r['levels'][j] is None:
   it['eligible']=False;it.setdefault('holds',[]).append('source_visual_tracking_or_view_hold' if held else 'audited_grade_unavailable')
  if previous!=r['levels'][j] or held:events.append(dict(id=r['id'],exercise=r['exercise'],item=it['item'],bank=it['bank'],old=previous,new=r['levels'][j],eligible=it['eligible'],note_ko=v.get('note_ko')))
 baseitems=copy.deepcopy(r['items']);levels=copy.deepcopy(r['levels'])
 def first(item):return next(((it,y) for it,y in zip(baseitems,levels) if it['item']==item), (None,None))
 def input_held(item):return '*' in v.get('input_holds',[]) or item in v.get('input_holds',[])
 if r['exercise']=='pushups':
  it,y=first('return_extension');res=sf.return_turn(x,q,frames,d['fps'],d['window_seconds'][1]*d['fps'])
  if input_held('return_extension'):res=dict(status='held',reason='source_visual_required_arm_tracking_or_view_hold')
  # This semantic arm head intentionally excludes trunk/support labels. It
  # does not claim full standard-pushup quality for knee-supported variants.
  add(r,'return_extension','observed_post_descent_return_v2',y,res,it)
 if r['exercise']=='ytw':
  variant='upright' if 'upright' in r.get('variant','') else 'hinged'
  for item,kind in [('y_reach','Y'),('t_reach','T'),('w_shape','W')]:
   it,y=first(item);res=sf.stage(x,frames,d['fps'],kind)
   if input_held(item):res=dict(status='held',reason='source_visual_required_arm_tracking_or_view_hold')
   # The prior compact stage contract is the closest paired baseline; it
   # chooses three observations, whereas this version requires five.
   prior=next((a for a in oldmap[r['id']]['items'] if a['item']==item and a['bank']==variant+'_compact_shape_v3'),None)
   if prior is None:prior=next((a for a in oldmap[r['id']]['items'] if a['item']==item),None)
   add(r,item,variant+'_stage_contiguous_v4',y,res,prior)
 if r['exercise']=='shoulder':
  it,y=first('frontal_trunk_hold');posture='seated' if 'seated' in r.get('variant','') or (it and it['bank']=='front_seated' and r.get('variant')!='standing_press') else 'standing'
  ticks=np.rint(frames/d['fps']*10).astype(int);view=r.get('view','front')
  if 'oblique' in (it or {}).get('bank',''):view='front_oblique'
  if len(ticks) and len(set(ticks))==len(ticks):res=body.extract_body(list(zip(ticks,x)),[int(ticks[0]),int(ticks[-1])],exercise_confirmed=y is not None,width=d['width'],height=d['height'],view=view)
  else:res=dict(status='held',reason='source_clock_10hz_slots_ambiguous')
  if input_held('frontal_trunk_hold'):res=dict(status='held',reason='source_visual_body_tracking_hold')
  add(r,'frontal_trunk_hold','four_joint_body_v2/front/'+posture,y,res,it)
 if r['exercise']=='plank' and '/recovered-observations/' in d['cache_path']:
  # Recompute the same units from the freshly verified source extraction.
  # Observability is item-specific; 0.58s can support alignment, never 3s continuity.
  a=plank_item_observation.assess(x,d['window_seconds'][1]-d['window_seconds'][0],[True]*4);ch=geometry.channels(x)
  for j,it in enumerate(r['items']):
   if it['bank']=='side_support_clearance_v1':
    c=held_geometry.plank_clearance(x);it.update(feature=c['feature'],eligible=c['eligible'] and r['levels'][j] is not None,holds=c['holds'],observation_reextracted=True)
   elif it['bank']=='side_video' and it['item'] in plank_item_observation.ITEMS:
    k=plank_item_observation.ITEMS.index(it['item']);ob=a['items'][k];curve=ch[ob['side']+'_'+plank_item_observation.CHANNELS[k]];good=np.isfinite(curve);f=np.percentile(curve[good],[5,10,25,50,75,90,95]).tolist() if good.any() else None
    it.update(feature=f,eligible=ob['eligible'] and r['levels'][j] is not None and f is not None,holds=ob['reasons'],observation_reextracted=True)
   elif it['bank']=='side_body_alignment_v3':
    f=sf.compact.side_body(x);it.update(feature=f,eligible=f is not None and r['levels'][j] is not None,holds=[] if f is not None else ['required_side_body_geometry_unobserved'],observation_reextracted=True)
   it.pop('rule_grade',None)
assert_training_allowed(rows)
# Reuse previously source-reviewed exact shoulder item windows without turning
# overlapping windows into independent people or recordings.
reused=json.loads((B/'output/shoulder-expanded-train-v3-20261025/training-inputs.frozen.json').read_text());assert_training_allowed(reused)
for d in reused:
 assert hashlib.sha256(Path(d['cache_path']).read_bytes()).hexdigest()==d['cache_sha256']
 item=d['item'];bank=('four_joint_body_v2/front/'+d['posture'] if item=='frontal_trunk_hold' else 'anatomical_lower_turn_v1/'+d['view']+'/'+d['posture'])
 r=dict(id='reused_exact_item_window/'+d['id'],exercise='shoulder',person=d['person'],group='reused_person:'+d['person'],family='existing_audited_shoulder_item_windows',split='train',sources=d['sources'],levels=[d['grade']],items=[dict(item=item,bank=bank,feature=d['features'],eligible=True,holds=[],observation_receipt=d)],origin='reused_existing_Train_item_window',source_window_seconds=d['feature_window_seconds'],overlap_not_independent=True)
 # De-duplicate exact same feature contract and source-clock window.
 key=(tuple(r['sources']),item,bank,tuple(d['all_context_native_frames']))
 r['deduplication_key']=hashlib.sha256(repr(key).encode()).hexdigest();rows.append(r)
 old.append(dict(**{k:v for k,v in r.items() if k not in ('items','levels')},items=[dict(item=item,bank=bank,feature=None,eligible=False,holds=['no_equivalent_parent_interval_feature'])],levels=[d['grade']]))
assert len({r['id'] for r in rows})==len(rows);assert_training_allowed(rows)
for name,value in [('score-inputs.frozen.json',rows),('paired-baseline-inputs.frozen.json',old),('label-change-receipts.json',events),('feature-observation-receipts.json',receipts),('reused-shoulder-item-windows.frozen.json',reused)]:
 p=O/name;assert not p.exists();p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
summary=dict(existing_interval_candidates=609,reused_precise_shoulder_item_windows=len(reused),reused_windows_are_not_independent_people_or_recordings=True,total_input_records=len(rows),visual_review_intervals=74,review_by_exercise=dict(collections.Counter(r['exercise'] for r in audit.values())),changed_grade_intervals=len({r['id'] for r in events if r['old']!=r['new']}),changed_grade_item_banks=sum(r['old']!=r['new'] for r in events),input_hold_intervals=sum(bool(r['input_holds']) for r in audit.values()),new_semantic_features=dict(collections.Counter((r['exercise']+'/'+r['item']) for r in receipts if r['eligible'])),historical_sources_relabelled=False,Test_used=False)
(O/'intake-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(summary)
