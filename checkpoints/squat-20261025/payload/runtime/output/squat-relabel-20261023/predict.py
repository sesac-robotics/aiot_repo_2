"""Isolated new-criteria candidate inference; never replaces the team bundle."""
from pathlib import Path
import hashlib
import joblib
import numpy as np
from criteria_features import observed_features
ROOT = Path(__file__).parent

def predict(image_xy):
    bundle = joblib.load(ROOT / 'candidate.joblib')
    assert hashlib.sha256((ROOT / 'labels.frozen.json').read_bytes()).hexdigest() == bundle['label_sha256']
    features = observed_features(image_xy)
    results = {}
    for key, head in bundle['heads'].items():
        f = features.get(key)
        if f is None:
            results[key] = {'status': 'unobserved', 'ai': None, 'rule': None}
            continue
        good, partial = head['boundaries']
        v = f['scalar']
        rule = 2 if (v >= good if head['higher'] else v <= good) else 1 if (v >= partial if head['higher'] else v <= partial) else 0
        results[key] = {'status': 'experimental_only', 'ai': float(np.clip(head['model'].predict([f['x']])[0], 0, 2)), 'rule': rule, 'coverage': f['coverage']}
    return {'items': results, 'total': None, 'promoted': False}

if __name__ == '__main__':
    import sys, json
    print(json.dumps(predict(np.load(sys.argv[1])['image_xy']), ensure_ascii=False, indent=2))
