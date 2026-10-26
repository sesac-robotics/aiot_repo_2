"""고정 라벨·관측·예약 분리 후 학습하고 재사용 개발 구간에서만 후보를 비교한다."""
import collections
import copy
import hashlib
import itertools
import json
import pathlib

import joblib
import numpy as np

from app.scoring.engine import ScoringEngine
from app.scoring.features import DEFINITIONS
from app.scoring.partition import audit_partitions
from app.scoring.rules import grades
from app.scoring.training import refit_heads

BASE=pathlib.Path('/home/jayden/SeSACPAI-Mid')
ROOT=BASE/'output/five-exercise-recovery-20261023'
OLD=BASE/'output/five-exercise-recovery-20261022'


def loss(predictions: list[dict]) -> dict:
    """동일 관측·참조 교집합에서 총점·항목·오류 항목 오차를 요약한다."""
    total=[abs(r['ai_score']-r['target_score']) for r in predictions if r['ai_score'] is not None and r['target_score'] is not None]
    errors=[abs(a-g) for r in predictions for a,g in zip(r['ai_levels'],r['levels']) if a is not None and g is not None and g<2]
    items=[abs(a-g) for r in predictions for a,g in zip(r['ai_levels'],r['levels']) if a is not None and g is not None]
    rule=[abs(r['rule_score']-r['target_score']) for r in predictions if r['rule_score'] is not None and r['target_score'] is not None]
    return dict(total_cases=len(total),AI_MAE=float(np.mean(total)) if total else None,Rule_MAE=float(np.mean(rule)) if rule else None,item_MAE=float(np.mean(items)),error_item_MAE=float(np.mean(errors)) if errors else None)


def evaluate(model: dict, rows: list[dict], baseline: dict) -> list[dict]:
    """기존 총점 가능 집합을 고정해 점수 채움 없이 두 모델을 비교한다."""
    out=[]
    for row in rows:
        old=baseline[row['id']];r=copy.deepcopy(old);values=[];rule_values=[]
        for i,item in enumerate(row['items']):
            key=i if row['exercise']=='ytw' else (row['exercise'],item['item'],item['bank'])
            head=model.get(key)
            if not item['eligible'] or item['feature'] is None or head is None:
                values.append(None);rule_values.append(None);continue
            estimator=head['model']
            value=old['ai_levels'][i] if isinstance(estimator,dict) else float(np.clip(estimator.predict([item['feature']])[0],0,2))
            values.append(value)
            rule_values.append(int(grades(np.asarray([item['feature']]),head['policy'],i)[0]) if row['exercise']=='ytw' else old['rule_levels'][i])
        weights=DEFINITIONS[row['exercise']][1]
        r['ai_levels']=values;r['rule_levels']=rule_values
        r['ai_score']=float(sum(w*g/2 for w,g in zip(weights,values))) if old['ai_score'] is not None and None not in values else None
        r['rule_score']=float(sum(w*g/2 for w,g in zip(weights,rule_values))) if old['rule_score'] is not None and None not in rule_values else None
        out.append(r)
    return out


def tune_projected_rules(model: dict, rows: list[dict], groups: dict) -> tuple[dict,list]:
    """직접 학습 등급만으로 화면 투영 규칙을 별도 보정한다. 최종 시험을 읽지 않는다."""
    candidate=copy.deepcopy(model);report=[]
    for i,head in candidate.items():
        bucket=[r for r in rows if r['split']=='train' and r['levels'][i] is not None and r['items'][i]['eligible']]
        normal={groups[r['id']] for r in bucket if r['levels'][i]==2};bad={groups[r['id']] for r in bucket if r['levels'][i]<2}
        if min(len(normal),len(bad))<2:continue
        x=np.asarray([r['items'][i]['feature'] for r in bucket]);y=np.asarray([r['levels'][i] for r in bucket]);c=collections.Counter((g,groups[r['id']]) for r,g in zip(bucket,y));gc=collections.Counter(g for g,_ in c);w=np.asarray([1/(gc[g]*c[g,groups[r['id']]]) for r,g in zip(bucket,y)])
        prior=tuple(head['policy']);options=[prior]
        if i==2:
            for bend,partial,hand,up in itertools.product((15,20,30,45),(5,8,15),(-.05,0,.025,.05),(-1,-.95,-.9,-.5,0)):
                if bend<=partial:continue
                p=list(prior);p[0]=bend;p[1]=partial;p[2]=hand;p[3]=min(hand-.05,p[3]);p[7]=up;p[8]=min(up,p[8]);options.append(tuple(p))
        elif i in (0,1):
            for a,b,c1,d in itertools.product((110,120,130,140,150) if i==0 else (65,70,75,80),(65,80,95,105) if i==0 else (40,50,60,70),(130,145,160),(80,100,120)):
                if a>b and c1>d:options.append((a,b,c1,d))
        else:
            for a,b,c1,d in itertools.product((1,1.5,2,2.5),(.5,.75,1,1.25),(.15,.25,.4),(.5,.75,1.0)):
                if a>b and c1<d:options.append((a,b,c1,d))
        def score(p):return float(np.average(abs(grades(x,p,i)-y),weights=w))
        selected=min(options,key=lambda p:(score(p),p!=prior,sum(abs(a-b) for a,b in zip(p,prior))))
        head['policy']=selected;head['rule_calibration']=dict(training_only=True,normal_groups=len(normal),error_groups=len(bad),selection_weighted_MAE=score(selected))
        report.append(dict(item=i,prior=list(prior),selected=list(selected),training_MAE_before=score(prior),training_MAE_after=score(selected)))
    return candidate,report


def main() -> None:
    """19개 직접 초안의 승인 항목을 보강해 5종 후보를 비교·보존한다."""
    source=json.loads((OLD/'held_round2/candidate_input_manifest.json').read_text());source=[r for r in source if r['exercise']!='squats']+json.loads((OLD/'squat_partial_top_trial/candidate_input_manifest.json').read_text())
    for r in source:r['label_origin']='direct_visual_draft'
    added=json.loads((ROOT/'new_feature_rows.json').read_text())['records'];rows=source+added
    reservation=json.loads((ROOT/'final_test_hashed_manifest.json').read_text());audit=audit_partitions(rows+reservation['records'],historical_people=reservation['historical_people'],historical_sources=reservation['historical_sources']);(ROOT/'post_label_partition_audit.json').write_text(json.dumps(audit,indent=2))
    groups={identifier:i for i,g in enumerate(audit['connected_groups']) for identifier in g['ids']}
    baseline={}
    for x in json.loads((OLD/'held_round2/comparison.json').read_text())['exercises']:
        if x['exercise']!='squats':baseline.update({r['id']:r for r in x['fixed_predictions']})
    sq=next(x for x in json.loads((OLD/'team_refinement/fixed_evaluation.json').read_text())['exercises'] if x['exercise']=='squats');baseline.update({r['id']:r for r in sq['fixed_predictions']})
    scorer=ScoringEngine();(ROOT/'models').mkdir(exist_ok=True);reports=[]
    for exercise in DEFINITIONS:
        subset=[r for r in rows if r['exercise']==exercise];validation=[r for r in subset if r['split']=='validation'];old=scorer.model(exercise)
        fitted,selection=refit_heads(old,subset,reserved_records=reservation['records'],historical_people=reservation['historical_people'],historical_sources=reservation['historical_sources'])
        raw_predictions=evaluate(fitted,validation,baseline);old_predictions=[baseline[r['id']] for r in validation]
        # 개발 검증에 따른 선택이며 독립 성능이 아니다. 항목별 알려진 오류의 악화를 막는다.
        accepted=copy.deepcopy(old);adopted_heads=[]
        for key,head in fitted.items():
            i=int(key) if isinstance(key,int) else DEFINITIONS[exercise][0].index(key[1])
            observations=[(new['ai_levels'][i],previous['ai_levels'][i],new['levels'][i]) for new,previous in zip(raw_predictions,old_predictions) if new['ai_levels'][i] is not None and previous['ai_levels'][i] is not None and new['levels'][i] is not None]
            if not observations:continue
            new_mae=np.mean([abs(a-g) for a,b,g in observations]);old_mae=np.mean([abs(b-g) for a,b,g in observations]);errors=[(a,b,g) for a,b,g in observations if g<2]
            error_ok=not errors or np.mean([abs(a-g) for a,b,g in errors])<=np.mean([abs(b-g) for a,b,g in errors])+1e-10
            if new_mae<old_mae-1e-8 and error_ok:
                accepted[key]=head;adopted_heads.append(str(key))
        accepted_predictions=evaluate(accepted,validation,baseline);before=loss(old_predictions);after=loss(accepted_predictions)
        improve=after['AI_MAE']<=before['AI_MAE']-1.0 and after['error_item_MAE']<=before['error_item_MAE']+1e-10
        if not improve:
            accepted=copy.deepcopy(old);accepted_predictions=old_predictions;adopted_heads=[]
        rule_adopted=False
        rule_report=[]
        if exercise=='ytw':
            rule_candidate,rule_report=tune_projected_rules(accepted,subset,groups);rule_predictions=evaluate(rule_candidate,validation,baseline)
            if loss(rule_predictions)['Rule_MAE']<after['Rule_MAE']-1e-8:
                # W 항목 오차도 악화시키지 않는 경우에만 채택한다.
                def w_error(preds):return np.mean([abs(r['rule_levels'][2]-r['levels'][2]) for r in preds if r['rule_levels'][2] is not None and r['levels'][2] is not None])
                if w_error(rule_predictions)<=w_error(accepted_predictions)+1e-10:accepted=rule_candidate;accepted_predictions=rule_predictions;rule_adopted=True
            joblib.dump(rule_candidate,ROOT/'models/ytw_rule_candidate.joblib')
        joblib.dump(fitted,ROOT/'models'/f'{exercise}_refit.joblib');joblib.dump(accepted,ROOT/'models'/f'{exercise}_accepted.joblib')
        record=dict(exercise=exercise,baseline=before,raw_refit=loss(raw_predictions),accepted=loss(accepted_predictions),ai_adopted=improve,rule_adopted=rule_adopted,adopted_heads=adopted_heads,training_selection=selection,rule_trial=rule_report,fixed_predictions=accepted_predictions,raw_predictions=raw_predictions,new_draft_rows=sum(r['exercise']==exercise for r in added),validation_is_reused_development=True,independent_final_test=False)
        reports.append(record);(ROOT/'comparison.json').write_text(json.dumps(dict(exercises=reports,completed=False),indent=2))
    (ROOT/'comparison.json').write_text(json.dumps(dict(exercises=reports,completed=True,final_test_opened=False,validation_informed_acceptance=True),indent=2))


if __name__=='__main__':main()
