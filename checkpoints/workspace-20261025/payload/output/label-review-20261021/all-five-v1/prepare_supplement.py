"""Y–T–W와 프레스의 새 참여자 영상을 직접 검토할 근거를 준비한다."""
import json
import hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from prepare_plank import ROOT,FONT
BASE=ROOT.parents[2]


def main():
    """추가 학습 후보 16개의 시간순 장면과 출처를 저장한다.

    Returns:
        None: supplement_manifest.json, supplement_*.jpg를 출력한다.
    Raises:
        RuntimeError: 영상 읽기 실패 시.
    """
    existing=json.loads((ROOT.parent/'other-three-v1/candidate_manifest.json').read_text());used={r['person'] for r in existing};pool=json.loads((BASE/'output/overnight-20261020/score_candidates.json').read_text());selected=[];rng=np.random.default_rng(311)
    for label,count in [('standing ytw - torso parallel to the ground, thumbs pointed up',4),('standing ytw - arms too low',2),('standing ytw - too fast',1),('standing ytw - too slow',1)]:
        candidates=[r for r in pool if r['exercise']=='ytw' and r['split']=='train' and label in r['source_labels']];rng.shuffle(candidates);added=0
        for r in candidates:
            if r['person'] in used:continue
            used.add(r['person']);selected.append(dict(r));added+=1
            if added==count:break
    shoulder=[r for r in pool if r['exercise']=='shoulder' and r['split']=='train' and '_front' in r['id'] and r['person'] not in used];people=sorted(set(r['person'] for r in shoulder))[:4];selected.extend(dict(r) for r in sorted(shoulder,key=lambda r:r['id']) if r['person'] in people)
    hashes=set()
    for n,r in enumerate(selected,1):
        sha=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest();r['video_sha256']=sha;r['duplicate_in_batch']=sha in hashes;hashes.add(sha);r['review_number']=n
        cap=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);fps=cap.get(5);count=int(cap.get(7));last=min(count-1,round(12*fps));indices=set(np.linspace(0,last,24).round().astype(int));frames=[]
        for idx in range(last+1):
            if not cap.grab():raise RuntimeError(r['id'])
            if idx in indices:
                ok,frame=cap.retrieve()
                if not ok:raise RuntimeError(r['id'])
                frames.append((idx,Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))))
        cap.release();canvas=Image.new('RGB',(1536,4*216+35),'#14212a');d=ImageDraw.Draw(canvas);d.text((5,4),f'SUP {n:02d} | {r["exercise"]} | TRAIN',font=FONT,fill='white')
        for j,(idx,im) in enumerate(frames):
            im.thumbnail((256,185));x=j%6*256;y=35+j//6*216;canvas.paste(im,(x+(256-im.width)//2,y));d.text((x+3,y+188),f'f{idx} {idx/fps:.2f}s',font=FONT,fill='white')
        evidence=ROOT/f'supplement_{n:02d}.jpg';canvas.save(evidence,quality=95);r['evidence']=dict(path=str(evidence),fps=fps,review_start=0,review_end=last,total_frames=count,sample_frames=sorted(int(i) for i in indices));print(n,r['id'],flush=True)
    (ROOT/'supplement_manifest.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2));print('새 참여자 후보',len(selected),flush=True)

if __name__=='__main__':main()
