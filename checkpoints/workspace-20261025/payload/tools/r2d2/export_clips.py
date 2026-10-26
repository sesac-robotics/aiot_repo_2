from pathlib import Path
import sys,subprocess,json,hashlib,concurrent.futures
sys.path.insert(0,str(Path(__file__).resolve().parent));from decisions import DECISIONS
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/r2d2-review-20261024';D=B/'data/r2d2-team-20261023';inventory=json.loads((O/'inventory.json').read_text());sources={r['id']:r for r in inventory}
def export(args):
 d,s=args;r=sources[d['video']];a=s['start'];b=min(s['end'],r['duration']);cid=f"{d['video']}_{int(a*1000):06}_{int(b*1000):06}";out=D/'clips'/d['exercise']/(cid+'.mp4');out.parent.mkdir(parents=True,exist_ok=True)
 if not out.exists():
  temp=out.with_name(out.stem+'.tmp.mp4')
  subprocess.run(['ffmpeg','-v','error','-threads','2','-ss',str(a),'-i',r['copy'],'-t',str(b-a),'-an','-vf','scale=min(720\,iw):-2','-c:v','libx264','-preset','ultrafast','-crf','18','-threads','2','-fps_mode','vfr','-movflags','+faststart','-y',str(temp)],check=True)
  temp.replace(out)
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=width,height','-of','json',str(out)]));duration=float(probe['format']['duration']);assert abs(duration-(b-a))<.15,(cid,duration,b-a)
 # Decode every frame of exported clips to detect corrupt outputs.
 subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(out),'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
 x=dict(id=cid,path=str(out),source_id=d['video'],source_sha256=r['sha256'],start=a,end=b,actual_duration=duration,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),width=probe['streams'][0]['width'],height=probe['streams'][0]['height'],audio_removed=True,all_frames_decoded=True);print(cid,flush=True);return x
args=[(d,s) for d in DECISIONS for s in d['segments']]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(export,args))
(D/'clip-manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print('CLIPS',len(rows),flush=True)
