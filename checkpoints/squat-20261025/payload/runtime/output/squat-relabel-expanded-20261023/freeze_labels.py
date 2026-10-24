"""Expanded direct visual draft; existing labels stay unchanged, no prediction-derived targets."""
import json,hashlib
from pathlib import Path
P=Path(__file__).parent;old=P.parent/'squat-relabel-20261023'
records=json.loads((old/'labels.frozen.json').read_text())['records'];ids={r['id'] for r in records}
# Explicitly fill only after direct review of each person's additions contact sheet.
reviewed={
 'Petra':{6:(2,1,2,2,2,None),7:(2,1,2,2,2,None),8:(2,1,2,2,0,None),9:(2,1,2,2,2,None),10:(2,1,2,2,2,None)},
 'Terka':{6:(2,1,2,2,2,None),7:(2,1,2,2,2,None),8:(2,1,2,2,None,None),9:(2,1,2,2,2,None),10:(2,1,2,2,2,None)},
}
reviewed['Vasil']={6:(2,2,2,2,2,None),7:(2,2,2,2,2,None),8:(0,2,2,2,2,None),9:(2,1,2,2,None,None),10:(2,2,2,2,2,None),11:(2,2,2,2,2,None)}
reviewed['Mima']={6:(2,1,2,2,2,None),7:(2,1,2,2,2,None),8:(2,1,2,2,2,None),9:(2,1,2,2,2,None),10:(2,1,2,2,2,None)}
keys=['knee_inward','descent_range','standing_return','trunk_balance','heel_support','spine_control']
for r in json.loads((P/'source-manifest.json').read_text()):
 if r['id'] in ids:continue
 n=int(r['id'].split('_')[1]);levels=dict(zip(keys,reviewed[r['person']][n]))
 evidence={k:'양 시점 시간순 표본 직접 검토; '+('표본만으로 확정 불가' if v is None else '뚜렷한 해당 오류 신호 없음' if v==2 else '부분 범위 또는 부분 신호' if v==1 else '뚜렷한 해당 오류 신호 확인') for k,v in levels.items()}
 evidence['descent_range']='측면 하단 허벅지 수평 도달' if levels['descent_range']==2 else '측면 하단 허벅지가 수평 위인 부분 하강; 개인에게 잘못된 운동이라는 뜻 아님'
 evidence['standing_return']='시작과 마지막 표본에서 선 자세로 복귀 확인'
 evidence['spine_control']='표본 이미지로 미세 허리 분절·빠른 반동 확정 불가'
 if r['person']=='Petra' and n==8:evidence['heel_support']='측면 하강 표본에서 뒤꿈치가 올라간 앞발 지지 확인'
 if r['person']=='Terka' and n==8:evidence['heel_support']='발·바닥 접촉 경계가 불명확해 들림 등급 확정하지 않음'
 records.append(dict(id=r['id'],person=r['person'],split=r['split'],recording_group='ex6_'+r['person'],sources=[v['sha256'] for v in r['views'].values()],levels=levels,evidence=evidence,label_origin='source_aware_direct_visual_draft',rule_generated_label=False,expert_validated=False,whole_normal_approved=False,review_frames_per_view=9,review_method='정면 cam1·측면 cam3 확대 시간순 표본; 연속 재생 아님',reference_goal='하강 범위는 허벅지 수평 참조의 영상상 기술, 개인 강제 깊이 아님',review_evidence=str(P/(r['person']+'-additions.jpg'))))
raw=json.dumps(dict(schema='squat-observation-checklist-v2',frozen_before_training=True,records=records),ensure_ascii=False,indent=2).encode();(P/'labels.frozen.json').write_bytes(raw);(P/'labels.sha256').write_text(hashlib.sha256(raw).hexdigest());print('frozen',len(records))
