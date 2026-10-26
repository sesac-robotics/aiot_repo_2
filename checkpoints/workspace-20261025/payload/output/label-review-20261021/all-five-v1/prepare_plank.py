"""추가 플랭크 원본의 장면을 시간순으로 추출한다. 점수를 자동 생성하지 않는다."""
import cv2
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)


def overview():
    """공개 출처 영상의 5초 간격 검토판과 시간 기록을 저장한다.

    Returns:
        None: overview_*.jpg와 source_timing.json을 출력한다.
    Raises:
        RuntimeError: 영상 열기·장면 읽기 실패 시.
    """
    source=ROOT/'sources/fitnessfaqs_plank.mp4';cap=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2]);fps=cap.get(5);count=int(cap.get(7));frames=[]
    if fps<=0:raise RuntimeError('프레임 속도 누락')
    for second in range(0,int(count/fps),5):
        cap.set(cv2.CAP_PROP_POS_MSEC,second*1000);ok,frame=cap.read()
        if not ok:raise RuntimeError(str(second))
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((320,180));frames.append((second,im))
    cap.release()
    for start in range(0,len(frames),24):
        part=frames[start:start+24];canvas=Image.new('RGB',(1920,4*210),'#14212a');draw=ImageDraw.Draw(canvas)
        for j,(t,im) in enumerate(part):
            x=j%6*320;y=j//6*210;canvas.paste(im,(x,y));draw.text((x+4,y+184),f'{t}s',font=FONT,fill='white')
        canvas.save(ROOT/f'overview_{start//24+1:02d}.jpg',quality=95)
    (ROOT/'source_timing.json').write_text(json.dumps(dict(path=str(source),fps=fps,total_frames=count,duration_seconds=count/fps,reviewed_sample_seconds=[t for t,_ in frames]),indent=2));print('플랭크 출처 시간순 선별판',len(frames))

if __name__=='__main__':overview()
