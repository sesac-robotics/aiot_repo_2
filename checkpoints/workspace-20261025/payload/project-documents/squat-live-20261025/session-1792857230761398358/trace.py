exec(open('/home/jayden/SeSACPAI-Mid/project-documents/squat-live-20261025/session-1792857230761398358/analyze.py').read().split('source=read_source()')[0].split('rows=[]')[0].split('for tick,pts in zip(ticks,xy):')[0])
t=MovementTracker('squats','front');trace=[]
for tick,pts in zip(ticks,xy):
 tick=int(tick);v=mask_frame_border(pts,w,h);hip=np.mean(v[[23,24]],axis=0);shoulder=np.mean(v[[11,12]],axis=0);scale=max(30.,float(np.linalg.norm(shoulder-hip)));base=t.base_hip;base_scale=t.base_scale
 before={k:getattr(t,k)for k in ['ready','bottom','returned','has_bottom','active','base_hip','base_scale','top_confirm','phase']}
 seg=t.push(tick,pts,width=w,height=h)
 if 60395<=tick<=60435:
  trace.append({'tick':tick,'seconds':(tick-int(ticks[0]))/10,'hip_y':float(hip[1]),'scale_now':scale,'fixed_baseline_drop':float((hip[1]-base)/base_scale)if base is not None else None,'before':before,'after':{k:getattr(t,k)for k in before},'completed':seg is not None,'reference_frames':t.standing_reference.get('frames')if isinstance(t.standing_reference,dict)else None})
out=B/'project-documents/squat-live-20261025/session-1792857230761398358';(out/'rep2-tick-trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2)+'\n')
for r in trace:
 if 60416<=r['tick']<=60429:print(r['tick'],round(r['hip_y'],3),round(r['fixed_baseline_drop'],5)if r['fixed_baseline_drop']is not None else None,{k:r['after'][k]for k in ['ready','bottom','returned','has_bottom','active','top_confirm']},r['after']['phase'],'complete',r['completed'])
