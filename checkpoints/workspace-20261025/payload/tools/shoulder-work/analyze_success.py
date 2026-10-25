from pathlib import Path
import sys,json,hashlib,collections
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');L=B/'output/live-webcam-20261023';sys.path.insert(0,str(L))
from shoulder_candidate import ShoulderTracker
S=L/'sessions/1792849624147364792';d=json.loads((S/'final-state.json').read_text());p=np.load(S/'pose.npz');t=ShoulderTracker();segments=[];phases=collections.Counter()
for tick,xy in zip(p['frame_numbers'],p['image_xy']):
 seg=t.push(int(tick),xy,int(p['width']),int(p['height']));phases[t.phase]+=1
 if seg:segments.append(seg)
rows=[]
for seg,r in zip(segments,d['repetitions']):
 score=r['score_100'];rows.append({'number':r['number'],'start_tick':seg[0][0],'end_tick':seg[-1][0],'samples':len(seg),'duration':(seg[-1][0]-seg[0][0])/10,'slot_coverage':len(seg)/(seg[-1][0]-seg[0][0]+1),'all_required_observed':bool(np.isfinite(np.array([x for _,x in seg])[:,[11,12,13,14,15,16,23,24]]).all()),'ai_subtotal':score['ai']['earned_subtotal'],'ai_assessed_weight':score['ai']['assessed_weight'],'ai_total':score['ai']['total'],'rule_total':score['rule']['total'],'items':score['items']})
report={'session':str(S),'scope':'deployed functional replay, no video ground truth','samples':len(p['frame_numbers']),'seconds':(int(p['frame_numbers'][-1])-int(p['frame_numbers'][0]))/10,'saved_count':d['movement']['count'],'replayed_count':t.count,'evaluation_rows':len(d['repetitions']),'segments_nonoverlapping':all(a[-1][0]<b[0][0] for a,b in zip(segments,segments[1:])),'phase_counts':dict(phases),'rows':rows,'rule_mean':float(np.mean([r['rule_total'] for r in rows])),'ai_subtotal_mean':float(np.mean([r['ai_subtotal'] for r in rows])),'sha256':{n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in ['pose.npz','state.json','final-state.json']},'video_present':bool(list(S.glob('*.mp4'))),'models_changed':False}
O=B/'output/shoulder-work-20261024/success-1792849624147364792';O.mkdir(exist_ok=True);(O/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['rows','phase_counts','sha256']},ensure_ascii=False));print(json.dumps([{k:r[k] for k in ['number','samples','duration','slot_coverage','ai_subtotal','rule_total']} for r in rows],ensure_ascii=False))
