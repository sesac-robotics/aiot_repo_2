"""정면 관측 근거에 한정한 안내. 교정 분류 모델의 예측으로 위장하지 않는다."""
import numpy as np
VERSION='front-coach-observation-v1'
SOURCES=[{'title':'Mayo Clinic: Squat exercise','url':'https://www.mayoclinic.org/healthy-lifestyle/fitness/multimedia/squat/vid-20084663'}, {'title':'NASM: Squat biomechanics','url':'https://www.nasm.org/resource-center/blog/training/biomechanics-of-the-squat'}]

def measures(x):
    """몸통과 양 무릎의 정면 투영 관측을 각각 반환한다."""
    if not np.isfinite(x[[11,12,23,24]]).all():return None
    s=x[[11,12]].mean(0);h=x[[23,24]].mean(0);scale=np.linalg.norm(h-s)
    hipwidth=abs(x[24,0]-x[23,0])
    if scale<30 or hipwidth<10:return None
    ratio=abs(x[26,0]-x[25,0])/hipwidth if np.isfinite(x[[25,26]]).all() else None
    return dict(tilt=float(np.degrees(np.arctan2(h[0]-s[0],h[1]-s[1]))),knee_ratio=ratio)

class LiveCoach:
    """준비 자세 기준선·0.5초 지속 조건을 사용하는 실험용 관측 안내."""
    def __init__(self):
        self.base=None;self.hits={'knee':0,'trunk':0};self.last_tick=None
    def push(self,tick,xy,ready):
        m=measures(xy)
        if self.last_tick is not None and tick-self.last_tick>2:
            self.base=None;self.hits={'knee':0,'trunk':0}
        self.last_tick=tick
        if m is None:
            self.hits={'knee':0,'trunk':0};self.base=None
            return dict(status='관측 부족',messages=['어깨·골반이 보이도록 구도를 확인하세요.'],version=VERSION)
        if self.base is None:
            if ready and m['knee_ratio'] is not None:self.base=m
            return dict(status='준비 자세 확인',messages=['편안히 선 자세를 잠시 유지하세요.'],version=VERSION)
        narrow=m['knee_ratio'] is not None and m['knee_ratio']<self.base['knee_ratio']*.75
        lean=abs(m['tilt']-self.base['tilt'])>10
        for k,condition in [('knee',narrow),('trunk',lean)]:self.hits[k]=self.hits[k]+1 if condition else 0
        messages=[]
        if self.hits['knee']>=5:messages.append('양 무릎 간격이 시작보다 좁아졌습니다. 무릎이 발 방향을 따라 움직이는지 확인하세요.')
        if self.hits['trunk']>=5:messages.append('몸통의 좌우 기울기가 시작보다 커졌습니다. 중심을 천천히 확인하세요.')
        if m['knee_ratio'] is None:messages.append('양 무릎 관측 부족 · 무릎 방향 안내는 보류합니다.')
        return dict(status='확인 안내' if messages else '뚜렷한 관측 경고 없음',messages=messages or ['현재 관측 범위에서 동작을 확인 중입니다.'],version=VERSION,validated=False,checks=item_statuses(live=True,knee_state=('관측 부족 · 무릎 간격 안내 보류' if m['knee_ratio'] is None else '간격 감소 신호 · 발 방향 확인 필요' if self.hits['knee']>=5 else '뚜렷한 간격 감소 신호 없음 · 발 정렬 미확인'),trunk_state='좌우 기울기 변화 관측' if self.hits['trunk']>=5 else '뚜렷한 좌우 기울기 변화 없음'),method='시작 기준 대비 25% 무릎 간격 감소 또는 10도 몸통 변화가 0.5초 지속된 실험 규칙')

def completed_feedback(xy,candidate):
    """완료 구간의 관측·점수를 근거로 안내하고 관측 불가와 정상은 구분한다."""
    ms=[measures(x) for x in xy];available=[m for m in ms if m is not None];messages=[];evidence={}
    if len(available)<.8*len(xy):messages.append('몸통 관측 부족 · 몸통 교정 판단 보류')
    else:
        tilts=np.array([m['tilt'] for m in available]);change=float(np.percentile(abs(tilts-np.median(tilts[:5])),90));evidence['trunk_tilt_change_deg']=round(change,2)
        if change>10:messages.append('몸통 좌우 기울기 변화가 컸습니다. 다음 반복에서 중심 이동을 확인하세요.')
    knees=[m['knee_ratio'] for m in available if m['knee_ratio'] is not None]
    if len(knees)<.8*len(xy):messages.append('양 무릎 관측 부족 · 무릎 간격 변화 판단 보류')
    elif len(knees)>=10:
        ratio=float(np.percentile(knees,10)/max(.01,np.median(knees[:5])));evidence['knee_spacing_relative_to_start']=round(ratio,3)
        if ratio<.75:messages.append('시작 대비 무릎 간격 감소가 관측됐습니다. 무릎과 발의 방향을 함께 확인하세요.')
    if candidate and not candidate.get('error'):
        if candidate['ai_depth_grade']<1:messages.append('하강 범위 추정이 작습니다. 구도와 동작 범위를 함께 확인하세요. 더 깊게 하라는 확정 지시는 아닙니다.')
        if candidate['return_height_grade']<2:messages.append('종료 높이가 시작과 달랐습니다. 다음 반복 전 편안한 선 자세로 돌아왔는지 확인하세요.')
    if not messages:messages=['정면 관측 범위에서 뚜렷한 교정 신호가 확인되지 않았습니다.']
    return dict(version=VERSION,messages=messages,evidence=evidence,checks=item_statuses(evidence,candidate),method='정면 투영 관측 규칙 + 실험 하강 추정',not_model_diagnosis=True,limitations=['무릎 간격 감소는 발 방향을 확인한 실제 무릎 안쪽 붕괴 진단이 아님','정확한 깊이·뒤꿈치·앞뒤 무릎 이동은 정면에서 확정하지 않음','관측 안내를 점수의 추가 감점으로 중복 적용하지 않음'])

def item_statuses(evidence=None,candidate=None,live=False,knee_state=None,trunk_state=None):
    """개별 관측 상태를 명시하고 불충분한 정보를 정상으로 채우지 않는다."""
    from squat_criteria import CRITERIA
    e=evidence or {};c=candidate or {};valid=c and not c.get('error')
    knee='관측 부족 · 발 방향 함께 확인 필요'
    if knee_state is not None:knee=knee_state
    elif 'knee_spacing_relative_to_start' in e:knee='간격 감소 신호 · 발 방향 확인 필요' if e['knee_spacing_relative_to_start']<.75 else '뚜렷한 간격 감소 신호 없음 · 발 정렬 미확인'
    trunk=trunk_state or ('좌우 기울기 변화 관측' if e.get('trunk_tilt_change_deg',0)>10 else '뚜렷한 좌우 기울기 변화 없음' if 'trunk_tilt_change_deg' in e else '관측 부족')
    states=[knee,('하강 범위 작게 추정 · 실제 깊이 미확정' if c.get('ai_depth_grade',2)<1 else '하강 범위 추정 제공 · 실제 깊이 미확정') if valid else '회차 완료 후 추정' if live else '추정 보류',('시작 높이 복귀 관측' if c['return_height_grade']==2 else '시작 높이와 차이 관측') if valid else '회차 완료 후 확인' if live else '관측 부족',trunk,'발 지지 자동 평가 미연결 · 발 관측 필요','정면에서 허리 정렬·미세 반동 미확정']
    return [dict(id=i['id'],title=i['title'],status=s,scope=i['scope']) for i,s in zip(CRITERIA,states)]
