"""지정 구간을 확대 추출한다. 점수는 만들지 않으며 입출력 구간 단위는 초이다."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)

def sheet(path, start, end, output):
    """초 구간을 24장으로 추출해 확대 검토판과 프레임 번호를 반환한다.

    Args:
        path: 로컬 영상 Path.
        start: 시작 시각, 초.
        end: 종료 시각, 초.
        output: 저장할 JPEG 경로.
    Returns:
        dict: fps, 시작·종료 프레임 및 근거 이미지 경로.
    Raises:
        RuntimeError: 지정 프레임을 읽을 수 없는 경우.
    """
    cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);fps=cap.get(5);indices=np.unique(np.linspace(round(start*fps),round(end*fps),24).round().astype(int));canvas=Image.new('RGB',(1600,6*250),'#14212a');draw=ImageDraw.Draw(canvas)
    for j,i in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,im=cap.read()
        if not ok:raise RuntimeError(str(path))
        im=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));im.thumbnail((400,220));x=j%4*400;y=j//4*250;canvas.paste(im,(x+(400-im.width)//2,y));draw.text((x+3,y+222),f'f{i} {i/fps:.3f}s',font=FONT,fill='white')
    cap.release();canvas.save(output,quality=95)
    return dict(start_frame=int(indices[0]),end_frame=int(indices[-1]),fps=fps,evidence=str(output),sample_frames=indices.tolist())

if __name__=='__main__':
    manifest=[]
    for n,(start,end) in enumerate([(10,13),(77,80),(94,97),(178,181)],1):
        r=sheet(ROOT/'sources/forearm_plank.mp4',start,end,ROOT/f'airforce_{n:02d}.jpg');r.update(video_path=str(ROOT/'sources/forearm_plank.mp4'),review_number=n);manifest.append(r)
    (ROOT/'airforce_segments.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
