"""Audit exact source-video duplicates before model evaluation."""
from pathlib import Path
from collections import defaultdict
import hashlib,json
BASE=Path(__file__).resolve().parent

def main():
    groups=defaultdict(list)
    for r in json.loads((BASE/'data/selected.json').read_text()):
        with open(r['video_path'],'rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        groups[digest].append(dict(video_id=r['video_id'],worker_id=r['worker_id'],source_split=r['source_split']))
    duplicates=[v for v in groups.values() if len(v)>1]
    report=dict(exact_duplicate_groups=duplicates,cross_worker_duplicate_groups=[v for v in duplicates if len({r['worker_id'] for r in v})>1])
    (BASE/'data/duplicate_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if report['cross_worker_duplicate_groups']:raise ValueError('Cross-worker identical clips require grouping before training.')

if __name__=='__main__':main()
