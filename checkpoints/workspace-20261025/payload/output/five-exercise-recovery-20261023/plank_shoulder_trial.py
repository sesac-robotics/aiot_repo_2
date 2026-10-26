"""같은 관절의 어깨 높이 특징을 추가해 지지 AI 후보만 별도 검증한다."""
import copy
import json
import pathlib
import runpy

import joblib
import numpy as np

from app.scoring.engine import ScoringEngine
plank_clearance = runpy.run_path(str(pathlib.Path(__file__).parent / "held_geometry_shoulder_experimental.py"))["plank_clearance"]
refit_heads = runpy.run_path(str(pathlib.Path(__file__).parent / "training_shoulder_experimental.py"))["refit_heads"]

BASE=pathlib.Path('/home/jayden/SeSACPAI-Mid')
ROOT=BASE/'output/five-exercise-recovery-20261023'
OLD=BASE/'output/five-exercise-recovery-20261022'


def main() -> None:
    """기존 관측 일치부터 확인하며 시험 예약 자료의 특징·등급을 열지 않는다."""
    functions=runpy.run_path(str(ROOT/'train_and_compare.py'))
    rows=[r for r in json.loads((OLD/'held_round2/candidate_input_manifest.json').read_text()) if r['exercise']=='plank']
    rows += [r for r in json.loads((ROOT/'new_feature_rows.json').read_text())['records'] if r['exercise']=='plank']
    for row in rows:row['label_origin']='direct_visual_draft'
    maps={r['id']:r['coordinates'] for r in json.loads((ROOT/'plank_sequences.json').read_text())['records']}
    diagnostics=[];missing=[];template=ScoringEngine().model('plank')
    previous_key=('plank','leg_and_body_support','side_support_clearance_v1');key=('plank','leg_and_body_support','side_support_clearance_v2')
    for row in rows:
        note=row['items'][2]
        if not note['eligible'] or note['bank']!=previous_key[2]:continue
        if row['id'] not in maps:missing.append(row['id']);note['eligible']=False;continue
        seq=np.load(maps[row['id']],allow_pickle=False);new=plank_clearance(seq,include_shoulder=True)
        if not new['eligible'] or not np.allclose(new['feature'][:22],note['feature'],atol=1e-5,equal_nan=False):
            raise ValueError('Reviewed old features and raw sequence differ: '+row['id'])
        note.update(bank=key[2],feature=new['feature'])
        diagnostics.append(dict(id=row['id'],grade=row['levels'][2],split=row['split'],shoulder_height_p10=new['feature'][23]))
    (ROOT/'plank_shoulder_features.json').write_text(json.dumps(dict(records=rows,diagnostics=diagnostics,missing_sequences=missing),indent=2))
    if any(r['id'] in missing and r['split']=='validation' for r in rows):raise ValueError('Validation coverage changed')
    reservation=json.loads((ROOT/'final_test_hashed_manifest.json').read_text())
    fitted,selection=refit_heads({key:template[previous_key]},rows,reserved_records=reservation['records'],historical_people=reservation['historical_people'],historical_sources=reservation['historical_sources'])
    candidate=copy.deepcopy(template);candidate.pop(previous_key);candidate.update(fitted)
    report=next(r for r in json.loads((OLD/'held_round2/comparison.json').read_text())['exercises'] if r['exercise']=='plank');baseline={r['id']:r for r in report['fixed_predictions']};validation=[r for r in rows if r['split']=='validation']
    predictions=functions['evaluate'](candidate,validation,baseline);before=functions['loss'](list(baseline.values()));after=functions['loss'](predictions)
    adopted=after['AI_MAE']<=before['AI_MAE']-1 and after['error_item_MAE']<=before['error_item_MAE']+1e-10
    result=dict(baseline=before,candidate=after,adopted=adopted,selection=selection,fixed_predictions=predictions,missing_sequences=missing,final_test_opened=False,validation_is_reused_development=True,new_features_are_joint_geometry_not_score_labels=True,rule_unchanged=True)
    joblib.dump(candidate,ROOT/'models/plank_shoulder_candidate.joblib',compress=3);(ROOT/'plank_shoulder_trial.json').write_text(json.dumps(result,indent=2));print(result['baseline'],result['candidate'],adopted)


if __name__=='__main__':main()
