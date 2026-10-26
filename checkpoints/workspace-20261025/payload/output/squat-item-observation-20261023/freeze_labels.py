"""Direct visual item labels frozen before fitting; no predictions used."""
import json,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent
# source ID: selected first complete interval, visible item labels, visual reason.
D={
'00241647':(0,98,None,2,2,'발목·서 있는 무릎은 잘리지만 어깨·골반의 좌우 안정과 시작 높이 복귀를 확인. 무릎 라벨은 없음.'),
'00232606':(0,145,2,None,2,'머리·상단 어깨가 잘려도 골반·양 무릎의 간격은 관찰 가능. 어깨 기준 몸통은 미평가.'),
'00082383':(0,83,2,2,2,'좁은 스탠스 자체를 오류로 삼지 않음. 첫 하강·복귀에서 뚜렷한 무릎 간격 붕괴·좌우 기울임 없음.'),
'00159159':(0,113,None,2,2,'하단이 잘리고 어두워 무릎은 미평가. 첫 하강·복귀의 몸통 좌우 안정은 관찰 가능.'),
'00188989':(0,107,None,None,None,'긴 옷이 골반·양 무릎을 가림. 머리·발목 때문이 아니라 필요한 관절 관계가 불명확해 해당 항목을 미라벨로 보존.'),
'00133938':(95,167,2,None,2,'상단이 잘리지만 골반·양 무릎의 좌우 간격은 확인. 어깨 기준선이 잘려 몸통 라벨은 없음.'),
'00071275':(0,117,2,2,2,'발목이 잘려도 어깨·골반·양 무릎이 보임. 무릎은 하강 중 바깥쪽으로 움직이며 뚜렷한 안쪽 모임 없음.'),
'00265473':(0,113,None,2,2,'무릎 아래가 잘려 무릎·발은 미평가. 몸통 좌우 안정과 첫 반복의 높이 복귀 확인.'),
'00156413':(0,117,None,2,2,'발목·하단 무릎 잘림과 별개로 어깨·골반의 좌우 변화와 높이 복귀 확인.'),
'00016136':(0,49,2,2,2,'첫 짧은 하강 후 시작 높이 복귀. 얕은 하강을 무릎·몸통 오류로 전이하지 않음.'),
'00021263':(0,79,2,2,2,'첫 하강·복귀 구간에서 무릎 간격과 몸통 좌우 안정 확인. 깊이는 별도 라벨 없음.'),
'00022867':(0,68,2,2,2,'첫 짧은 하강·복귀에서 뚜렷한 무릎 모임·몸통 좌우 변화 없음. 깊이 정상 여부는 미평가.'),
'00031937':(0,58,2,2,2,'첫 하강·복귀 구간에서 무릎·몸통 좌우 안정 확인. 구도와 깊이 라벨을 혼합하지 않음.'),
'00032830':(0,78,2,2,2,'첫 완결 하강·상승에서 양 무릎이 바깥쪽을 유지하고 몸통 좌우 안정. 같은 qevd_1463과 동일 사람 그룹.'),
'00042076':(0,117,None,2,2,'전후 숙임은 좌우 몸통 오류로 변환하지 않음. 좁은 무릎 겹침으로 무릎 간격 정상 여부는 미확정.'),
'00045823':(0,69,2,2,2,'첫 짧은 하강 후 기립. 몸통 좌우 안정과 무릎 간격은 평가하되 깊이는 미라벨.'),
'00050396':(0,79,2,None,2,'머리·어깨 상단이 잘려도 골반·양 무릎의 간격 및 골반 높이 복귀 확인. 몸통은 미라벨.'),
}
rows=[]
for r in json.loads((O/'review-manifest.json').read_text()):
 a,b,k,t,ret,why=D[r['id']]
 rows.append(dict(**r,first_frame=a,last_frame=b,levels={'knee_inward':k,'trunk_balance':t,'standing_return':ret,'descent_range':None,'heel_support':None,'spine_control':None},label_origin='independent_direct_chronological_visual_draft',review_method='16 chronological raw-frame samples, not continuous playback',expert_validated=False,model_outputs_used=False,rule_generated=False,whole_normal_approved=False,reason=why))
raw=json.dumps(dict(criterion_version='front-observation100-consistent-v1',frozen_before_training=True,records=rows),ensure_ascii=False,indent=2).encode();(O/'labels.frozen.json').write_bytes(raw);(O/'labels.sha256').write_text(hashlib.sha256(raw).hexdigest());print(len(rows),sum(any(v is not None for v in r['levels'].values()) for r in rows))
