from pathlib import Path
import json,sys,copy,numpy as np
sys.dont_write_bytecode=True
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/four-exercise-source-phase-v2-20261025';sys.path.insert(0,str(B/'five-exercise-comparison'));import extract,audit
audit.OUT=O/'resource-state';audit.OUT.mkdir(exist_ok=True)
plans=json.loads((O/'legacy-extraction-plan.frozen.json').read_text());receipts=[]
for r in plans:
 spec=dict(id=r['id'],video_sha256=r['extraction_source_sha256'],required_views=r['required_views'],segment=r['segment']);rec=extract.extract_view(spec,'primary',r['source_path'],O/'recovered-observations');z=np.load(rec['cache']);fs=z['frame_numbers'];r=copy.deepcopy(r);r.update(cache_path=rec['cache'],cache_fingerprint=rec['fingerprint'],cache_indices=list(range(len(fs))),actual_native_frames=fs.tolist(),cache_context_frames=fs.tolist());receipts.append(r);(O/'legacy-recovered-inputs.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n');print(r['id'],len(fs),flush=True)
