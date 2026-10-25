from pathlib import Path
from fractions import Fraction
import json,sys,collections,hashlib
import numpy as np,cv2
B=Path('/home/jayden/SeSACPAI-Mid');sys.path.insert(0,str(B/'tools/shoulder-work'));from phase_features_v2 import extract
sys.path.insert(0,str(B/'tools/error-enrichment'));from split_guard import assert_training_allowed,POLICY
policy=json.loads(POLICY.read_text());team=json.loads((B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json').read_text())['labels'];inventory={r['id']:r for r in json.loads((B/'output/r2d2-review-20261024/inventory.json').read_text())};out=[]
for r in team:
 if r['exercise']!='shoulder' or policy['assignments'][r['source_id']]!='train':continue
 assert_training_allowed([dict(id=r['id'],person=r['person'],sources=[r['source_sha256']],split='train')])
 src=inventory[r['source_id']];p=B/'output/integrated-five-20261024/team-observations'/r['source_id']/'side.npz';meta=json.loads(p.with_suffix('.json').read_text());assert meta['config']['source_sha256']==r['source_sha256'];z=np.load(p);assert str(z['fingerprint'])==meta['fingerprint'];times=z['frame_numbers']/float(Fraction(src['fps']));ticks=np.rint(times*10).astype(int);mask=(times>=max(0,r['start']-.6))&(times<r['end']+.4)
 cap=cv2.VideoCapture(src['copy']);w=int(cap.get(3));h=int(cap.get(4));cap.release()
 pairs=list(zip(ticks[mask],z['image_xy'][mask]));f=extract(pairs,w,h)
 if f['status']=='features_ready' and not r['start']<=f['top_tick']/10<r['end']:f={'status':'held','reason':'selected_top_outside_original_label_interval'}
 out.append({'id':r['id'],'person':r['person'],'split':'train','source_sha256':r['source_sha256'],'dimensions':[w,h],'original_interval':[r['start'],r['end']],'input_context_seconds':[-.6,.4],'features':f,'existing_item_grades':{i['id']:i['grade'] if i['training_mask'] else None for i in r['items']},'label_created':False})
O=B/'output/shoulder-work-20261024/phase-model-v1';(O/'team-train-extraction-v2.json').write_text(json.dumps({'records':out,'status_counts':dict(collections.Counter(r['features'].get('reason',r['features']['status']) for r in out)),'fit_performed':False,'all_source_roles_preserved':True},ensure_ascii=False,indent=2)+'\n');print(collections.Counter(r['features'].get('reason',r['features']['status']) for r in out))
