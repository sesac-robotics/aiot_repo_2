"""One versioned frontal observation contract for points, comparison and coaching.

Preserves anatomical core100 in legacy_score_100. These frontal proxies are
explicitly named; they do not establish knee extension, foot alignment or safety.
Standing return optionally uses the versioned learned standing-state classifier.
Missing values are never imputed or rescaled.
"""
from copy import deepcopy
from math import isfinite
from rubric100 import assessment as legacy_assessment

VERSION = 'front-observation100-consistent-v1'
ITEMS = [
    ('descent_range', '하강 범위 추정', 30),
    ('standing_return', '서기 높이 복귀', 20),
    ('knee_inward', '무릎 간격 유지', 30),
    ('trunk_balance', '몸통 좌우 안정', 20),
]
SCOPES = {
    'descent_range': '정면 관절 특징으로 하강 범위를 추정합니다. 실제 깊이 측정이 아닙니다.',
    'standing_return': '하강 후 서기 복귀: AI는 초기 기립 상태 유사도, Rule Base는 골반 높이 차를 봅니다. 무릎 완전 신전 평가는 아닙니다.',
    'knee_inward': '시작 대비 양 무릎 간격 변화입니다. 발끝 방향·발 지지의 종합 평가가 아닙니다.',
    'trunk_balance': '시작 대비 몸통·어깨의 좌우 기울기입니다. 전후 허리 정렬 평가는 아닙니다.',
}
CUES = {
    'descent_range': '하강 범위 추정과 구도를 함께 확인하세요.',
    'standing_return': '동작 후 시작했던 선 자세의 높이로 돌아왔는지 확인하세요.',
    'knee_inward': '무릎 간격 변화와 무릎·발이 향하는 방향을 함께 확인하세요.',
    'trunk_balance': '몸통의 좌우 기울기와 중심 이동을 확인하세요.',
}


def valid_grade(value):
    return isinstance(value, (int, float)) and isfinite(value) and 0 <= value <= 2


def assessment(row, phase='complete'):
    """Read the same-window comparison payload; never borrow a legacy grade."""
    comparison = row.get('method_comparison') or {}
    source = {i['item']: i for i in comparison.get('item_comparisons', [])}
    items = []
    for key, title, weight in ITEMS:
        src = source.get(key, {})
        usable = src.get('status') in ('close', 'different', 'insufficient_error_labels', 'state_pending')
        a, b = (src.get('ai_grade'), src.get('rule_grade')) if usable else (None, None)
        reasons = [SCOPES[key]]
        if src.get('summary') and key == 'standing_return':
            reasons.append(src['summary'])
        if not usable:
            a = b = None
            reasons.append('관측·특징 또는 같은 기준의 평가 결과가 부족합니다.')
        if key == 'standing_return':
            if not src.get('standing_state_model'):
                a = None
                reasons.append('학습된 기립 상태 모델의 현재 구간 출력이 없어 AI 점수는 미평가입니다.')
            else:
                reasons.append('AI는 초기 기립 상태에 대한 학습 신뢰도×20으로 평가합니다. Rule Base의 높이 경계 점수를 복사하지 않습니다.')
            if phase != 'complete':
                a = b = None
                reasons.append('진행 중에는 종료 높이 복귀를 평가하지 않습니다.')
        a = a if valid_grade(a) else None
        b = b if valid_grade(b) else None
        ap = round(weight * a / 2, 1) if a is not None else None
        bp = round(weight * b / 2, 1) if b is not None else None
        delta = round(ap - bp, 1) if ap is not None and bp is not None else None
        # Thresholds are existing 0/1/2 label categories, not newly fitted cutoffs.
        disagreement = a is not None and b is not None and (a < 1) != (b < 1)
        warns = [name for name, g in [('AI Model', a), ('Rule Base', b)] if g is not None and g < 1]
        ai_return_partial=key=='standing_return' and a is not None and src.get('standing_state_model') and not src.get('standing_confirmed')
        partial = (b is not None and b == 1) or ai_return_partial
        if warns or partial:
            message = (('·'.join(warns) + '에서 낮게 추정했습니다. ') if warns else 'AI Model에서 기립 상태 확인이 부족합니다. ' if ai_return_partial else 'Rule Base에서 부분 충족으로 평가했습니다. ') + CUES[key]
            if disagreement:
                message += ' 두 방식의 판단이 달라 실제 오류 여부는 확인이 필요합니다.'
        elif a is None and b is None:
            message = title + ('는 회차 완료 후 확인합니다.' if key == 'standing_return' and phase != 'complete' else '는 관측·평가 근거가 부족합니다.')
        else:
            message = title + '에서 낮은 등급 신호가 없습니다.'
        items.append(dict(id=key, title=title, weight=weight, ai_points=ap,
                          rule_points=bp, estimated=True, experimental=True,
                          reasons=reasons, criterion_version=VERSION,
                          comparison=dict(difference_points=delta,
                              explanation=((f'AI Model {ap:.1f}/{weight}점 · Rule Base {bp:.1f}/{weight}점. ' + ('AI는 기립 상태 신뢰도, Rule Base는 골반 높이 경계를 사용합니다. ' if key=='standing_return' else 'AI는 학습된 여러 특징을, Rule Base는 같은 라벨로 보정한 단일 특징 경계를 사용합니다. ') + src.get('model_evidence', {}).get('explanation', '') + (' 규칙 경계 근처입니다.' if src.get('rule_evidence', {}).get('near_boundary') else '')) if delta is not None else '같은 기준의 양쪽 점수가 있을 때만 차이를 계산합니다.'),
                              automatic=True, gpt_used=False),
                          evidence=deepcopy(src),
                          coaching=dict(message=message, attention=bool(warns or partial),
                                        disagreement=disagreement,
                                        unavailable=a is None and b is None)))
    methods = {}
    for method in ('ai', 'rule'):
        known = [i for i in items if i[method + '_points'] is not None]
        full = len(known) == len(items) and phase == 'complete'
        subtotal = round(sum(i[method + '_points'] for i in known), 1)
        methods[method] = dict(total=subtotal if full else None, maximum=100,
            earned_subtotal=subtotal, assessed_weight=sum(i['weight'] for i in known),
            full_score=full, estimated=True,
            status='정면 관측 추정 · 검증 중' if full else '진행 중 · 잠정 항목 평가' if phase != 'complete' else '일부 항목 미평가')
    return dict(version=VERSION, phase=phase, items=items, **methods,
        expert_validated=False, total_renormalized=False,
        scope='정면 관측 기준 v1 · 하강 추정 30 / 서기 높이 복귀 20 / 무릎 간격 유지 30 / 몸통 좌우 안정 20. 기존 관절 신전·발 정렬 종합 기준과 구분하며, 검증 중인 추정 점수입니다.')


def coaching(score):
    attention = [i for i in score['items'] if i['coaching']['attention']]
    missing = [i['title'] for i in score['items'] if i['coaching']['unavailable']]
    messages = [i['coaching']['message'] for i in attention]
    if not messages:
        messages = ['평가된 정면 관측 항목에서 낮은 등급 신호가 없습니다.']
    if missing:
        messages.append('미평가: ' + ', '.join(missing) + '.')
    return dict(version=VERSION, phase=score['phase'], messages=messages,
        checks=[dict(id=i['id'], title=i['title'], **i['coaching']) for i in score['items']],
        automatic=True, gpt_used=False, validated=False,
        scope='점수와 같은 구간·항목·판정 결과에서 생성한 관측 안내입니다.')


def apply(row, exercise='squats', view='front', phase='complete'):
    result = deepcopy(row)
    if exercise != 'squats' or view != 'front':
        result['score_100'] = legacy_assessment(result, exercise)
        return result
    result.setdefault('legacy_score_100', legacy_assessment(result, exercise))
    result.setdefault('legacy_coaching', deepcopy(result.get('coaching')))
    result.setdefault('legacy_method_totals', {key:deepcopy(result.get(key)) for key in ('ai','rule')})
    result['score_100'] = assessment(result, phase)
    for method in ('ai','rule'):
        summary=result['score_100'][method]
        result[method]=dict(earned=summary['earned_subtotal'],maximum=summary['assessed_weight'],
            full_score=summary['full_score'],total=summary['total'],criterion_version=VERSION)
    result['coaching'] = coaching(result['score_100'])
    return result
