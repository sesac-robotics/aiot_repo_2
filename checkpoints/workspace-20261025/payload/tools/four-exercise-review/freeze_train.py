from pathlib import Path
import json,copy,hashlib,collections,sys
sys.dont_write_bytecode=True
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/four-exercise-review-train-v1-20261025';P=B/'output/squat-fixed-reference-combined-v3-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'));from split_guard import assert_training_allowed
rows=json.loads((P/'score-inputs.frozen.json').read_text());plan=json.loads((O/'expanded-review-plan.frozen.json').read_text());renders=json.loads((O/'expanded-render-receipts.json').read_text());labels=json.loads((O/'labels-audit-v1.frozen.json').read_text());changes={65:{'overhead_reach':2},66:{'overhead_reach':2},67:{'return_extension':None},68:{'support':0},78:{'trunk_line':1,'leg_and_body_support':1,'support_continuity':1},86:{'y_reach':1},87:{'y_reach':1}}
for n in range(70,78):changes[n]={'trunk_line':0,'leg_and_body_support':0}
notes={64:'첫 반복의 팔 상단이 굽힌 부분 도달에 머묾. 이 원본의 모든 반복을 정상으로 바꾸지 않음.',65:'9.6~10.0초 양팔이 거의 펴진 머리 위 도달 확인. 상단1을2로 수정; 일부 시차1 유지.',66:'25.4~25.6초 머리 위 양팔 도달 확인. 상단1을2로 수정; 일부 시차1 유지.',67:'초반 상단 유지 후 하강하고 구간 끝에서 다시 상승 중. 복귀 정상 확정 불가 null; 추가 지지 혼재1 유지.',68:'무릎·하체 바닥 추가 접촉이 하강/상승에 걸쳐 명확. 지지0 추가.',69:'같은 원본의 양팔 완전 도달·복귀 정상 대조, 기존2 유지.',78:'80.0초 바닥 추가 접촉에서 80.6초 이후 목표 지지로 전환. 전환 혼재이므로 정렬/지지/지속1.',85:'Y 상단은 머리 위로 좁게 모이는 부분1, T누락0, 양손을 올린 W2 구분. 기존 라벨 유지.',86:'머리 위로 좁게 모이는 I 형태가 유지되어 Y부분1. 기존2 수정; T부분1/W누락0 유지.',87:'이 구간에서 Y는 좁은 모임과 짧은 대각선 통과, 부분1. 비대칭 W1 및 몸통 보류 유지.'}
for n in range(70,78):notes[n]='이 구간 원본 프레임에서 하체·무릎 바닥 추가 접촉과 처진 정렬 직접 확인. 정상2를 정렬0/지지0으로 수정; 나쁜 자세 지속2는 별도 유지.'
for n in range(79,85):notes[n]='같은 Navy 원본의 후반 구간: 무릎·하체가 바닥에서 떨어지고 목표 지지를 유지. 앞 구간 오류를 원본 전체로 확장하지 않고 기존 정상2 유지.'
for k,r in enumerate(plan):
 n=64+k;labels.append(dict(index=n,id=r['id'],exercise=r['exercise'],person=r['person'],sources=r['sources'],split='train',source_path=r['source_path'],window_seconds=r['window_seconds'],review_note_ko=notes[n],grade_overrides=changes.get(n,{}),input_holds=[],reviewer='Codex_visual_source_review_not_expert_certification',evidence=[renders[k]['page']],actual_reviewed_source_frames=renders[k]['source_frames'],coverage='12_uniform_actual_cache_observations; not_every_video_frame',old_grades=[dict(item=a,grade=b,bank=c) for a,b,c in zip(r['previous_items'],r['previous_levels'],r['previous_banks'])]))
# Reuse the earlier audited shoulder body labels only when the exact full clip window matches.
prior=json.loads((B/'output/shoulder-expanded-train-v3-20261025/visual-labels.frozen.json').read_text())
prior_map={'r2d2_'+r['original_annotation_id']:r for r in prior['team_clips']};events=[];train=[];audit={r['id']:r for r in labels}
for parent in rows:
 if parent['exercise']=='squats' or parent['split']!='train':continue
 r=copy.deepcopy(parent);v=audit.get(r['id']);p=prior_map.get(r['id']) if r['exercise']=='shoulder' else None
 for j,item in enumerate(r['items']):
  prev=r['levels'][j];grade=prev;why=[]
  if p and item['item']=='frontal_trunk_hold':
   grade=p['body_grade'];why.append('reused_exact_clip_body_audit_v3')
  if v and item['item'] in v['grade_overrides']:grade=v['grade_overrides'][item['item']];why.append(v['review_note_ko'])
  if v and ('*' in v['input_holds'] or item['item'] in v['input_holds']):
   item['eligible']=False;item.setdefault('holds',[]).append('source_visual_confirmed_tracking_misconnection');why.append(v['review_note_ko'])
  if grade is None and prev is not None:
   item['eligible']=False;item.setdefault('holds',[]).append('audited_unobservable_or_incomplete_window')
  if grade!=prev or why:events.append(dict(id=r['id'],exercise=r['exercise'],item=item['item'],bank=item['bank'],old=prev,new=grade,eligible=item['eligible'],reasons=why))
  r['levels'][j]=grade
 r['audit_version']='four-exercise-Train-review-v2-20261025';train.append(r)
assert_training_allowed(train)
for name,v in [('labels-audit-v2.frozen.json',labels),('score-inputs.frozen.json',train),('label-change-receipts.json',events)]:
 p=O/name;assert not p.exists();p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
summary=dict(candidates_by_exercise=dict(collections.Counter(r['exercise'] for r in train)),new_visual_review_intervals=dict(collections.Counter(r['exercise'] for r in labels)),new_grade_changed_unique_intervals=len({r['id'] for r in events if r['old']!=r['new']}),new_grade_changed_item_banks=sum(r['old']!=r['new'] for r in events),prior_body_label_reuse_intervals=len(set(prior_map)&{r['id'] for r in train}),new_input_hold_intervals=len({r['id'] for r in labels if r['input_holds']}),Test_used=False,validation_fit=False,squats_touched=False)
(O/'intake-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(summary)
