"""Post-freeze side-view diagnostic, separate from byte-identical original Test."""
import json,sys,datetime,os
from fractions import Fraction
from pathlib import Path
import numpy as np,joblib
B=Path(__file__).resolve().parents[2];O=Path(os.environ.get('SIDE_RETURN_FINAL',str(B/'output/squat-side-return-enriched-20261024')))
sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'tools/integrated-five'),str(B/'tools/paired-evaluation')]
from common import sha
from team_features import describe,height_features
import side_height
from thresholds import predict,transform_input
read=lambda p:json.loads(p.read_text());models=joblib.load(O/'score-candidates.joblib');rules=read(O/'rules.frozen.json');key=('squats','standing_height_return','height_v4');head=models['heads'][key];rule=next(h for h in rules['heads'] if tuple(h['key'])==key)
before=sha(O/'test-inputs.frozen.json');model_sha=sha(O/'score-candidates.joblib');rules_sha=sha(O/'rules.frozen.json');assert (O/'model-freeze-before-test.json').exists()
protocol=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='supplemental historically exposed diagnostic side-view observations, not the original comparison denominator',source='V15',source_role='test',reference_window=[0,.22],reference_review_sheet_sha256=sha(O/'diagnostic-Test-V15-reference.jpg'),reference_rule='upright initial pose verified from original frames; exclude initial calibration context from first scored feature interval, no future return used',model_sha256=model_sha,rules_sha256=rules_sha,model_or_threshold_changes_allowed=False,existing_Test_modified=False,new_independent_Test=False)
p=O/'supplemental-side-Test-protocol.frozen.json';assert not p.exists();p.write_text(json.dumps(protocol,indent=2))
labels={r['id']:r for r in read(B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json')['labels']};inv=next(r for r in read(B/'output/r2d2-review-20261024/inventory.json') if r['id']=='V15');cp=B/'output/integrated-five-20261024/team-observations/V15/side.npz';z=np.load(cp);t=z['frame_numbers']/float(Fraction(inv['fps']));ref=side_height.reference(z['image_xy'][(t>=0)&(t<=.22)],height_features);out=[]
for row in read(O/'test-inputs.frozen.json'):
    if row['family']!='R2D2' or labels[row['id'][5:]]['source_id']!='V15':continue
    d=labels[row['id'][5:]];start=max(d['start'],.22);note=describe('squats',z['image_xy'][(t>=start)&(t<d['end'])],'side',d['end']-start,ref,1080,1920,baseline_verified=True,baseline_end=.22,segment_start=start)[1];record=dict(id=row['id'],person=row['person'],split='test',source_sha256=d['source_sha256'],source_interval=[d['start'],d['end']],feature_interval=[start,d['end']],note=note,target=next(i['grade'] for i in d['items'] if i['id']=='standing_height_return'))
    if note['eligible']:
        f=note['feature'];record.update(ai_prediction=float(np.clip(head['model'].predict([f])[0],min(head['grade_support']),max(head['grade_support']))),rule_prediction=float(predict(rule['rule'],transform_input('height_v4','standing_height_return',f))[0]),ai_candidate_gate_passed=head['candidate_gate_passed'])
    out.append(record)
assert len(out)==3 and before==sha(O/'test-inputs.frozen.json') and model_sha==sha(O/'score-candidates.joblib') and rules_sha==sha(O/'rules.frozen.json')
(O/'supplemental-side-Test-results.frozen.json').write_text(json.dumps(dict(records=out,recovered=sum(r['note']['eligible'] for r in out),original_Test_unchanged=True,model_unchanged=True,rules_unchanged=True,used_for_training_or_selection=False,not_independent_final_Test=True,normal_only_single_person_does_not_validate_error_grades=True),indent=2));print('SUPPLEMENTAL_SIDE',sum(r['note']['eligible'] for r in out),'of3')
