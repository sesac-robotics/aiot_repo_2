"""플랭크 전신·근접 구간을 직접 판정하기 위한 확대 근거를 준비한다."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from prepare_plank import ROOT,FONT
SEGMENTS=[(0.5,3.5),(14,17),(24,27),(29,32),(59,62),(64,67),(74,77),(79,82),(119,122),(129,132),(249,252),(429,432)]


def main():
    """각 구간의 16개 장면과 10FPS 접촉판을 저장한다.

    Returns:
        None: plank_case_*.jpg, plank_dense_*.jpg, plank_segments.json 출력.
    Raises:
        RuntimeError: 구간을 디코딩하지 못하는 경우.
    """
    source=ROOT/'sources/fitnessfaqs_plank.mp4';manifest=[]
    for n,(start,end) in enumerate(SEGMENTS,1):
        cap=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);fps=cap.get(5);a=round(start*fps);b=round(end*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,a);frames=[]
        for idx in range(a,b+1):
            ok,frame=cap.read()
            if not ok:raise RuntimeError(str(idx))
            if (idx-a)%max(1,round(fps/10))==0 or idx==b:frames.append((idx,Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))))
        cap.release()
        for kind,selected,cols,width,height in [('case',[frames[i] for i in np.linspace(0,len(frames)-1,16).round().astype(int)],4,384,240),('dense',frames,6,256,170)]:
            canvas=Image.new('RGB',(cols*width,((len(selected)+cols-1)//cols)*height),'#14212a');d=ImageDraw.Draw(canvas)
            for j,(idx,im) in enumerate(selected):
                im=im.copy();im.thumbnail((width,height-30));x=j%cols*width;y=j//cols*height;canvas.paste(im,(x,y));d.text((x+3,y+height-25),f'P{n:02d} {idx/fps:.3f}s',font=FONT,fill='white')
            canvas.save(ROOT/f'plank_{kind}_{n:02d}.jpg',quality=95)
        manifest.append(dict(review_number=n,path=str(source),fps=fps,start_frame=a,end_frame=b,sample_frames=[idx for idx,_ in frames],evidence=str(ROOT/f'plank_case_{n:02d}.jpg'),dense_evidence=str(ROOT/f'plank_dense_{n:02d}.jpg')))
    (ROOT/'plank_segments.json').write_text(json.dumps(manifest,indent=2));print('플랭크 구간 12개 근거 준비')

if __name__=='__main__':main()
