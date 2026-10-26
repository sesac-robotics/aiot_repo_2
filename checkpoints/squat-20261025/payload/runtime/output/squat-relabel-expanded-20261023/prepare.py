import json,shutil
from pathlib import Path
B=Path('/home/jayden/SeSACPAI-Mid');P=Path(__file__).parent;old=B/'output/squat-relabel-20261023'
prev=json.loads((old/'source-manifest.json').read_text());allrows=json.loads((B/'mediapipe-pilot/correction/sample-manifest.json').read_text());out=list(prev)
for person in ['Petra','Terka','Vasil','Mima']:
 base=next(r for r in prev if r['person']==person);known={(r['first_frame'],r['last_frame']) for r in prev if r['person']==person};n=5
 for r in allrows:
  if r['person']!=person or r['camera']!='2' or r['orientation']!='0' or (r['first_frame'],r['last_frame']) in known:continue
  n+=1;out.append(dict(base,id=f'{person}_{n:02d}',first_frame=r['first_frame'],last_frame=r['last_frame'],provider_label_candidate_only=r['label']))
for f in ['criteria_features.py','train.py','predict.py','test_candidate.py']:
 shutil.copy(old/f,P/f)
for r in prev:
 shutil.copytree(old/'observations'/r['id'],P/'observations'/r['id'],dirs_exist_ok=True)
(P/'source-manifest.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
s=(old/'prepare_dense.py').read_text();(P/'prepare_dense.py').write_text(s)
s=(old/'extract_front.py').read_text().replace("O=B/'output/squat-relabel-20261023'","O=Path(__file__).parent").replace("p.mkdir(parents=True,exist_ok=True)","p.mkdir(parents=True,exist_ok=True)\n if (p/'front.npz').exists():continue")
(P/'extract_front.py').write_text(s)
print(len(out))
