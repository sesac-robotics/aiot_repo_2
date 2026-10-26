"""기존 학습 관절의 점수 동일성과 모델 지문을 검사한다. 정확도 시험은 아니다."""
import hashlib
import json
from pathlib import Path
import numpy as np
from app.scoring.engine import ScoringEngine
from app.scoring.media import load_landmarks

ROOT=Path('/home/jayden/SeSACPAI-Mid/output/five-exercise-recovery-20261023')
BASE=ROOT.parent/'five-exercise-recovery-20261022/team_portability'
engine=ScoringEngine()
failures=[]
rows=json.loads((BASE/'cases.json').read_text())['records']
for row in rows:
    actual=engine.score(row['exercise'],load_landmarks(Path(row['landmarks'])),**row['options'])
    for key,expected in row['expected'].items():
        a=([item['ai_grade'] for item in actual['items']] if key=='ai_levels' else [item['rule_grade'] for item in actual['items']] if key=='rule_levels' else actual[key])
        def equal(x,y):
            if x is None or y is None:return x is y
            if isinstance(x,list):return len(x)==len(y) and all(equal(m,n) for m,n in zip(x,y))
            return bool(np.isclose(x,y,rtol=0,atol=1e-10))
        if not equal(a,expected):failures.append(dict(id=row['id'],field=key))
report=dict(cases=len(rows),identical=len(rows)-len({r['id'] for r in failures}),failures=failures,training_cases_not_accuracy_test=True,independent_final_test=False,model_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in engine.bundle.glob('*.joblib')})
(ROOT/'release_parity.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='model_hashes'}))
if failures:raise SystemExit(1)
