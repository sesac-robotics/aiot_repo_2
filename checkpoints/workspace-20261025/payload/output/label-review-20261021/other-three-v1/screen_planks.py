"""몸통·다리가 보이는 플랭크 촬영 방향을 사람이 선별할 표본판을 만든다."""
import copy
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from prepare import BASE,ROOT,FONT


def main():
    """추가 참여자의 중간 장면을 추출한다. 점수나 정답은 만들지 않는다.

    Returns:
        None: screen_manifest.json과 screen_*.jpg를 출력한다.
    Raises:
        RuntimeError: 프레임 디코딩 실패 시.
    """
    existing=json.loads((ROOT/'candidate_manifest.json').read_text());used={r['person'] for r in existing};source=json.loads((BASE/'output/overnight-20261020/score_candidates.json').read_text());pool=[r for r in source if r['exercise']=='plank' and r['split']=='train'];order=np.random.default_rng(71).permutation(len(pool));picked=[]
    for i in order:
        r=pool[int(i)]
        if r['person'] in used:continue
        used.add(r['person']);picked.append(copy.deepcopy(r))
        if len(picked)==120:break
    for n,r in enumerate(picked,1):
        cap=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);fps=cap.get(5);idx=int(cap.get(7))//2;cap.set(cv2.CAP_PROP_POS_FRAMES,idx);ok,frame=cap.read();cap.release()
        if not ok:raise RuntimeError(r['id'])
        r['screen_number']=n;r['screen_frame']=idx;r['screen_time']=idx/fps
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((240,160));canvas=Image.new('RGB',(240,190),'#14212a');canvas.paste(im,((240-im.width)//2,0));ImageDraw.Draw(canvas).text((5,165),f'S{n:03d} {idx/fps:.2f}s',font=FONT,fill='white');canvas.save(ROOT/f'screen_case_{n:03d}.jpg')
    for start in range(0,len(picked),24):
        canvas=Image.new('RGB',(6*240,4*190),'#14212a')
        for j in range(min(24,len(picked)-start)):canvas.paste(Image.open(ROOT/f'screen_case_{start+j+1:03d}.jpg'),(j%6*240,j//6*190))
        canvas.save(ROOT/f'screen_{start//24+1:02d}.jpg',quality=95)
    (ROOT/'screen_manifest.json').write_text(json.dumps(picked,ensure_ascii=False,indent=2));print('추가 참여자 120명 촬영 방향 선별판 준비')

if __name__=='__main__':main()
