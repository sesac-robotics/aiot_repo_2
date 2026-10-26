"""Original 30/20/30/20 contract. Missing items never become 0, full marks, or renormalized totals."""
from copy import deepcopy
ITEMS=[('descent_range','목표 하강 범위',30),('return_extension','상승·서기 복귀',20),('knee_foot','무릎·발 정렬',30),('trunk_balance','몸통 좌우 안정',20)]
VERSION='squat-core100-observation-v3'

def assessment(row,exercise='squats'):
    source={r['id']:r for r in row.get('items',[])}
    if exercise!='squats':
        defs=[(r['id'],r['id'],r['weight']) for r in row.get('items',[])]
    else:defs=ITEMS
    results=[];candidate=row.get('front_candidate') or {};cmp=row.get('method_comparison') or {}
    for key,title,weight in defs:
        original=source.get(key,{})
        a=original.get('ai_grade');b=original.get('rule_grade');reason=[];estimated=False
        if original.get('single_grade_training'):a=None;reason.append('AI는 정상 한 등급만 학습해 오류 판별이 미검증입니다.')
        if exercise=='squats':
            if key=='descent_range' and cmp.get('status') in ('different','close') and cmp.get('ai_points') is not None:
                # Preserved candidate trained on synchronized side-depth visual labels.
                # It estimates the original target; it is never sold as actual measured depth.
                a=cmp['ai_grade'];b=cmp['rule_grade'];estimated=True
                reason.append('동기화 측면 깊이 라벨로 학습한 정면 추정입니다. 목표 범위 일치도이며 개인에게 강요하는 깊이·안전성 점수가 아닙니다.')
            if key=='return_extension':
                # Height-based checklist is a diagnostic; not equivalent to knee extension.
                if a is None:reason.append('복귀 오류 AI 라벨·검증 부족. 골반 높이 복귀를 관절 신전 점수로 대체하지 않습니다.')
                if b is None:reason.append('이 정면 구간에서 실제 관절 신전 기준을 확인하지 못했습니다.')
            if key=='knee_foot' and (a is None or b is None):reason.append('발 방향·지지와 무릎 정렬의 합성 기준을 확인하지 못했습니다. 다른 항목·횟수 평가는 계속합니다.')
            if key=='trunk_balance':reason.append('정면의 좌우 안정 평가입니다. 전후 허리 정렬까지 판정하지 않습니다.')
        r=dict(id=key,title=title,weight=weight,ai_points=None if a is None else round(weight*a/2,1),rule_points=None if b is None else round(weight*b/2,1),estimated=estimated,reasons=list(dict.fromkeys(reason)),comparison=None)
        if a is not None and b is not None:
            delta=round(r['ai_points']-r['rule_points'],1)
            explanation=cmp.get('summary') if key=='descent_range' else f'같은 항목·배점에서 AI Model과 Rule Base의 차이는 {delta:+.1f}점입니다. 학습된 특징 조합과 규칙 경계가 다릅니다.'
            r['comparison']=dict(difference_points=delta,explanation=explanation,automatic=True,gpt_used=False)
        else:r['comparison']=dict(difference_points=None,explanation='양쪽이 같은 기준으로 평가된 경우에만 점수 차이를 계산합니다.',automatic=True,gpt_used=False)
        results.append(r)
    methods={}
    for method in ('ai','rule'):
        known=[r for r in results if r[method+'_points'] is not None]
        complete=len(known)==len(results) and bool(results)
        earned=round(sum(r[method+'_points'] for r in known),1)
        methods[method]=dict(total=earned if complete else None,maximum=100,earned_subtotal=earned,assessed_weight=sum(r['weight'] for r in known),full_score=complete,estimated=any(r['estimated'] for r in known),status='정면 추정 포함' if complete and any(r['estimated'] for r in known) else '전체 항목 평가' if complete else '일부 항목 미평가')
    return dict(version=VERSION,items=results,**methods,expert_validated=False,total_renormalized=False,scope='프로젝트 맨몸 스쿼트 목표 범위·정렬·복귀·좌우 안정; 100점은 의학적으로 검증된 안전 점수가 아님')
