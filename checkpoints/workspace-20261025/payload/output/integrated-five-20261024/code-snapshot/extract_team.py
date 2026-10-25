from pathlib import Path
import sys,json,cv2
B=Path('/home/jayden/SeSACPAI-Mid');sys.path.insert(0,str(B/'five-exercise-comparison'))
import extract,audit
O=B/'output/integrated-five-20261024';receipts=json.loads((O/'team-extraction.json').read_text()) if (O/'team-extraction.json').exists() else [];done={r['id'] for r in receipts}
for r in json.loads((B/'output/r2d2-review-20261024/inventory.json').read_text()):
 if r['id'] in done:continue
 cap=cv2.VideoCapture(r['copy']);fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
 row=dict(id=r['id'],video_sha256=r['sha256'],front_sha256=r['sha256'],required_views=['side' if int(r['id'][1:]) in [3,8,9,11,14,15,16,19] or 28<=int(r['id'][1:])<=36 or int(r['id'][1:])>=45 else 'front'],segment=dict(start_frame=0,end_frame=count-2,fps=fps))
 rec=extract.extract_view(row,'side',r['copy'],O/'team-observations');receipts.append(dict(id=r['id'],extraction=rec));audit.save_json(O/'team-extraction.json',receipts);print(r['id'],rec.get('frames'),flush=True)
