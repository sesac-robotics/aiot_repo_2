"""대표 동작의 단일 시도 구간을 확대하여 직접 등급 판정을 돕는다."""
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw
from prepare import ROOT,FONT
REQUESTS={17:(0,100),19:(0,117),20:(0,85),21:(0,119),23:(23,117),26:(39,110),27:(30,136),29:(70,110),30:(70,110),31:(16,55),32:(0,39),33:(0,31),34:(8,39),35:(39,63),36:(8,39),37:(8,47),38:(16,55),39:(16,55),40:(8,39)}


def main():
    """확대판과 시간순 원본 프레임 근거를 저장한다.

    Returns:
        None: detail_*.jpg와 dense_*.jpg를 출력한다.
    Raises:
        RuntimeError: 지정 구간의 프레임을 읽지 못할 경우.
    """
    for n,(a,b) in REQUESTS.items():
        r=json.loads((ROOT/f'{n:03d}.json').read_text());cap=cv2.VideoCapture(r['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);cap.set(cv2.CAP_PROP_POS_FRAMES,a);images=[]
        for k in range(a,b+1):
            ok,f=cap.read()
            if not ok:raise RuntimeError(r['id'])
            images.append((k,Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))))
        cap.release();selected=np.linspace(0,len(images)-1,12).round().astype(int);canvas=Image.new('RGB',(1536,3*280),'#14212a');d=ImageDraw.Draw(canvas)
        for j,k in enumerate(selected):
            idx,im=images[k];im=im.copy();im.thumbnail((384,250));x=j%4*384;y=j//4*280;canvas.paste(im,(x+(384-im.width)//2,y));d.text((x+3,y+252),f'{n:03d} f{idx} {idx/r["evidence"]["fps"]:.2f}s',font=FONT,fill='white')
        canvas.save(ROOT/f'detail_{n:03d}.jpg',quality=95)
        # 영상 전체 재생으로 표시하지 않고, 이 구간의 모든 원본 프레임을 판으로 보존한다.
        for start in range(0,len(images),48):
            part=images[start:start+48];canvas=Image.new('RGB',(1536,((len(part)+5)//6)*216),'#14212a');d=ImageDraw.Draw(canvas)
            for j,(idx,im) in enumerate(part):
                im=im.copy();im.thumbnail((256,185));x=j%6*256;y=j//6*216;canvas.paste(im,(x+(256-im.width)//2,y));d.text((x+3,y+188),f'f{idx} {idx/r["evidence"]["fps"]:.2f}s',font=FONT,fill='white')
            canvas.save(ROOT/f'dense_{n:03d}_{start//48+1}.jpg',quality=95)
        print('확대 근거',n,flush=True)

if __name__=='__main__':main()
