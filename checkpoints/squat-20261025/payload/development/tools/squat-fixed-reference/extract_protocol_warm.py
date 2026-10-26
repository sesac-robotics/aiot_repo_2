"""Fixed past-only tracking warmup; target windows/labels are unchanged."""
import datetime,json,sys
from pathlib import Path
B=Path(__file__).resolve().parents[2];sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'five-exercise-comparison')]
from common import sha
import extract
P=B/'output/shoulder-common-support-audit-20261024/full-review-v1/protocol-model-v1'
O=P.parent/'protocol-model-v2'
def main():
 O.mkdir(exist_ok=True)
 p=O/'plan.frozen.json';assert not p.exists();p.write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   parent=str(P),label_sha256=sha(P/'labels.frozen.json'),hypothesis='Cold start on a short raised-arm window may prevent tracking acquisition.',
   past_only_warmup_seconds=3,original_target_windows_unchanged=True,feature_samples_only_inside_original_label=True,
   observation_gate_unchanged=True,no_model_or_prediction_based_selection=True,final_Test_opened=False,production_changes=False),indent=2))
 (O/'labels.frozen.json').write_bytes((P/'labels.frozen.json').read_bytes());receipts=[]
 for r in json.loads((P/'labels.frozen.json').read_text())['labels']:
  fps=15.;row=dict(id=r['id'],video_sha256=r['source_sha256'],required_views=['side'],
     segment=dict(start_frame=round(max(0,r['start']-3)*fps),end_frame=round(r['end']*fps),fps=fps))
  rec=extract.extract_view(row,'side',r['path'],O/'observations');receipts.append(dict(id=r['id'],extraction=rec))
  (O/'extraction-progress.json').write_text(json.dumps(receipts,indent=2));print('WARM_PROTOCOL',r['id'],rec['frames'],flush=True)
 p=O/'extraction.frozen.json';assert not p.exists();p.write_text(json.dumps(receipts,indent=2));print('COMPLETE33',flush=True)
if __name__=='__main__':main()
