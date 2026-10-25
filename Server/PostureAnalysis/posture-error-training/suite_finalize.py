"""Materialize an exercise only when every selected video has finished extraction."""
import argparse,json
import numpy as np
from suite_config import SUITE,TASKS,CONFIG_VERSION

def finalize(exercise):
    rows=json.loads((SUITE/exercise/'selected.json').read_text());quality=json.loads((SUITE/'pose_quality.json').read_text())
    pending=[r['video_id'] for r in rows if not (SUITE/'cache'/f'{r["video_id"]}.npz').exists() and quality.get(r['video_id'],{}).get('status') not in ['excluded_pose_quality','failed']]
    if pending:raise RuntimeError(f'{exercise}: {len(pending)} videos not finished')
    usable=[r for r in rows if (SUITE/'cache'/f'{r["video_id"]}.npz').exists()];xs=[]
    for r in usable:
        with np.load(SUITE/'cache'/f'{r["video_id"]}.npz') as cache:
            assert str(cache['config_version'])==CONFIG_VERSION
            xs.append(cache['X'])
    np.savez_compressed(SUITE/exercise/'data.npz',X=np.stack(xs),y=np.array([r['target'] for r in usable],np.float32),groups=np.array([r['worker_id'] for r in usable]),ids=np.array([r['video_id'] for r in usable]),source_split=np.array([r['source_split'] for r in usable]))
    print(f'{exercise}: finalized {len(usable)} / {len(rows)} videos')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--exercise',choices=list(TASKS),required=True);a=p.parse_args();finalize(a.exercise)
