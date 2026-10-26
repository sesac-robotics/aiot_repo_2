from pathlib import Path
import json,csv,hashlib,collections,sys
from decisions import DECISIONS
from rubric_anchors import ANCHORS
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/r2d2-review-20261024';D=B/'data/r2d2-team-20261023';L=D/'annotations';L.mkdir(exist_ok=True)
inv=json.loads((O/'inventory.json').read_text());src={r['id']:r for r in inv}
R={
'squats':(['descent_range','standing_height_return','knee_gap','frontal_trunk_stability'],['하강 범위(정면은 대리 관찰)','서기 높이 복귀','무릎 간격 유지','정면 몸통 안정'],[30,20,30,20]),
'shoulder':(['overhead_reach','lower_return','bilateral_timing','frontal_trunk_stability'],['머리 위 도달','하단 복귀','양팔 동시성','정면 몸통 안정'],[30,25,25,20]),
'pushups':(['trunk_alignment','descent_range','extension_return','support'],['몸통 정렬','가슴 하강','팔 펴기 복귀','손·발 지지'],[40,30,20,10]),
'plank':(['trunk_alignment','elbow_position','support','continuity'],['몸통 정렬','팔꿈치 위치','전완·발 지지','구간 내 지속'],[40,25,20,15]),
'ytw':(['Y_shape','T_shape','W_shape','upright_trunk_stability'],['Y 형태','T 형태','W 형태','서 있는 몸통 안정'],[30,25,25,20])}
names={'squats':'스쿼트','shoulder':'앉은 양팔 프레스','pushups':'푸시업','plank':'플랭크','ytw':'서서 Y–T–W'}
rubrics={'version':'r2d2-visual-draft-v1','status':'non_expert_visual_draft','grades':{'0':'명확한 미충족','1':'부분 충족','2':'관찰 기준 충족','null':'가림·시점·변형으로 평가 불가'},'scoring':'항목 배점 × 등급 / 2. 모든 항목 관찰 및 완전한 시도일 때만 100점 만점 총점 산출. 관찰 항목만으로 100점 환산 금지.','limits':['임상·전문가 판정 또는 AI 출력이 아님','4fps 시간순 장면 검토, 경계 정밀도 약 0.25초','정면 스쿼트 깊이는 측면 관절각 판정과 동일하지 않음','하중·통증·근활성·가려진 접촉·견갑 세부 판정 없음','YTW는 upright 변형이며 기존 bent-over 자료와 분리','숄더는 실제 중량 덤벨이 없는 앉은 프레스 시범','3초 플랭크 창의 점수는 총 버티기 체력 점수가 아님'],'exercises':{e:{'name_ko':names[e],'variant':{'ytw':'upright_ytw','shoulder':'seated_unloaded_bilateral_press','plank':'forearm_feet_plank'}.get(e,e),'items':[dict(id=i,name_ko=n,weight=w) for i,n,w in zip(*r)]} for e,r in R.items()},'references':[{'url':'https://www.acefitness.org/resources/everyone/exercise-library/41/push-up/','scope':'기본 푸시업 동작 참고. 점수 배점은 프로젝트 정의.'},{'url':'https://www.acefitness.org/resources/everyone/exercise-library/32/front-plank/','scope':'전완 플랭크 기본 형태 참고. 수치 임계값을 임상 기준으로 주장하지 않음.'},{'url':'https://www.acefitness.org/continuing-education/prosource/september-2014/4972/dynamite-delts-ace-research-identifies-top-shoulder-exercises/','scope':'숄더 프레스 동작 참고, 무부하 촬영과 하중 운동 구분.'}]}
for ex in rubrics['exercises'].values():
 for item in ex['items']:
  item['grade_anchors_0_1_2']=ANCHORS[item['id']]
rows=[]
for d in DECISIONS:
 r=src[d['video']];ids,ns,weights=R[d['exercise']]
 for s in d['segments']:
  a=s['start'];b=min(s['end'],r['duration']);cid=f"{d['video']}_{int(a*1000):06}_{int(b*1000):06}";grades=s['levels'];complete=s.get('complete_cycle',True)
  observed=sum(w for w,g in zip(weights,grades) if g is not None);earned=sum(w*g/2 for w,g in zip(weights,grades) if g is not None)
  quality='clear_fault' if 0 in grades else 'partial_or_fault' if 1 in grades else 'insufficient_observation' if None in grades or not complete else 'observed_criteria_met'
  variant=rubrics['exercises'][d['exercise']]['variant']
  if 'straight_arm_variant' in d['tags']:variant='straight_arm_plank'
  if 'kneeling_variant' in d['tags']:variant='kneeling_plank'
  rows.append(dict(id=cid,source_id=d['video'],source_sha256=r['sha256'],person=d['person'],exercise=d['exercise'],variant=variant,view=d['view'],start=a,end=b,complete_cycle=complete,quality=quality,total_score=earned if observed==100 and complete else None,observed_points=earned,observed_max=observed,items=[dict(id=i,name_ko=n,weight=w,grade=g,points=None if g is None else w*g/2,training_mask=g is not None) for i,n,w,g in zip(ids,ns,weights,grades)],fault_tags=d['tags'],evidence_note_ko=d['note_ko'],evidence_scope='aggregate_observation_over_segment_not_precise_fault_onset',evidence_sheets=[str(p.relative_to(B)) for p in sorted(O.glob(d['video']+'-dense-*.jpg'))],clip=str((D/'clips'/d['exercise']/(cid+'.mp4')).relative_to(B)),annotation_status='visual_draft_requires_independent_review',review_method='chronological_4fps_contact_sheets',quality_training_candidate=complete and variant not in ['straight_arm_plank','kneeling_plank'],person_assignment='provisional_visual_continuity'))
rows.sort(key=lambda x:(x['source_id'],x['start']));assert len({r['id'] for r in rows})==len(rows)
unscored=[]
for v,r in src.items():
 rs=[x for x in rows if x['source_id']==v];assert rs,v;last=0
 for x in rs:
  assert 0<=x['start']<x['end']<=r['duration'];assert x['start']>=last-1e-6,(v,last,x)
  if x['start']>last:unscored.append(dict(source_id=v,start=last,end=x['start'],reason='unscored_transition_or_truncated_or_short_remainder'))
  last=x['end']
 if last<r['duration']:unscored.append(dict(source_id=v,start=last,end=r['duration'],reason='unscored_transition_or_truncated_or_short_remainder'))
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
write(L/'rubrics.json',rubrics);write(L/'labels.frozen.json',{'version':rubrics['version'],'training_run':None,'labels':rows});write(L/'unscored-intervals.json',unscored);write(L/'sources.json',inv)
folds=[]
for p in ['P01','P02','P03','P04']:
 train=[x['id'] for x in rows if x['person']!=p and x['quality_training_candidate']];val=[x['id'] for x in rows if x['person']==p and x['quality_training_candidate']];assert not set(train)&set(val)
 folds.append(dict(held_out_person=p,train=train,validation=val))
write(L/'split-manifest.json',{'scheme':'leave_one_person_out_4_fold','identity_status':'provisional_visual_assignment','policy':'Same person and all correlated clips remain on one side. No random frame split. Four people, not hundreds of independent participants. Validation is not an untouched test set.','folds':folds})
with (L/'labels.csv').open('w',newline='',encoding='utf-8-sig') as f:
 fields=['id','source_id','person','exercise','variant','start','end','complete_cycle','quality','total_score','observed_points','observed_max','fault_tags','evidence_note_ko','clip'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for x in rows:w.writerow({k:';'.join(x[k]) if k=='fault_tags' else x[k] for k in fields})
write(O/'report-data.json',{'labels':rows,'rubrics':rubrics,'sources':inv,'counts':dict(collections.Counter(x['exercise'] for x in rows))})
print(json.dumps({'clips':len(rows),'exercises':dict(collections.Counter(x['exercise'] for x in rows)),'quality':dict(collections.Counter(x['quality'] for x in rows)),'unscored_intervals':len(unscored)},ensure_ascii=False))
