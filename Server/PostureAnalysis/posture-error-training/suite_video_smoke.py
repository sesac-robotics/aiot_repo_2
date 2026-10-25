"""Decode one actual held-out video per new exercise through the live inference CLI."""
from pathlib import Path
import json,os,subprocess,sys
from suite_config import BASE,SUITE,TASKS

def main():
    results={}
    for exercise in TASKS:
        rows={r['video_id']:r for r in json.loads((SUITE/exercise/'selected.json').read_text())};splits=json.loads((SUITE/exercise/'outputs/splits.json').read_text());vid=splits['test']['video_ids'][0];source=rows[vid]['video_path']
        process=subprocess.run([sys.executable,str(BASE/'predict_exercise.py'),'--exercise',exercise,'--source',source,'--no-window'],capture_output=True,text=True,check=True)
        log=SUITE/exercise/'outputs/video_smoke.log';log.write_text(process.stdout+'\nSTDERR\n'+process.stderr)
        payloads=[]
        for line in process.stdout.splitlines():
            try:payloads.append(json.loads(line))
            except json.JSONDecodeError:pass
        final=payloads[-1]
        any_prediction=any(r.get('status')=='ok' or r.get('last_result',{}).get('status')=='ok' for r in payloads)
        assert final['processed_frames']>0 and any_prediction,(exercise,final)
        results[exercise]=dict(video_id=vid,processed_frames=final['processed_frames'],at_least_one_prediction=True,last_result=final['last_result'])
        print(exercise+': real video inference passed',flush=True)
    (SUITE/'video_smoke_results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')

if __name__=='__main__':main()
