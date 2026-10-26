"""기존 점수 초안의 반복 구간을 재검토할 근거와 라벨 분포를 만든다.

영상에서 정답을 자동 생성하지 않는다. 모든 출력은 실험 폴더에 저장한다.
"""
import json
from pathlib import Path
from collections import Counter
import cv2
cv2.setNumThreads(2)
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'expanded'
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)


def frames(path, first, last, count):
    """구간 내 지정 수의 프레임과 원본 번호를 반환한다.

    Args:
        path: 읽을 영상의 절대 경로.
        first: 첫 프레임 번호, 0부터 시작.
        last: 마지막 프레임 번호, 포함.
        count: 균등 표본 수, 장 단위.
    Returns:
        list: (프레임 번호, RGB 이미지) 목록.
    Raises:
        RuntimeError: 영상 프레임을 읽을 수 없는 경우.
    """
    cap = cv2.VideoCapture(str(path)); result = []
    for i in np.unique(np.linspace(first, last, count).round().astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, frame = cap.read()
        if not ok: raise RuntimeError(f'{path}: {i}')
        # REHAB26 원본은 인물이 보이는 아래쪽 영역을 확대한다.
        if 'rehab26' in str(path):
            h, w = frame.shape[:2]; frame = frame[int(h*.30):int(h*.98), int(w*.10):int(w*.98)]
        result.append((int(i), Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    cap.release(); return result


def main():
    """초안 전체의 재검토 목록·요약판을 저장한다.

    Returns:
        None: manifest.json, inventory.json, panel_*.jpg를 출력한다.
    Raises:
        RuntimeError: 영상 또는 대응 메타데이터가 누락된 경우.
    """
    rows = []; metadata = {json.loads(p.read_text())['id']: json.loads(p.read_text()) for p in OLD.glob('[0-9][0-9][0-9].json')}
    for file in ['pushups_labels.json', 'squats_labels.json']:
        for label in json.loads((OLD/file).read_text()):
            if label['score'] is None: continue
            m = metadata[label['id']]; seg = label['segment']; n = m['review_number']
            row = dict(label, review_number=n, audit_status='pending_direct_review')
            selected = frames(label['video_path'], seg['start_frame'], seg['end_frame'], 8)
            if label['exercise'] == 'squats':
                front = json.loads((OLD/'squat_front'/f'{n:03d}.json').read_text())
                row['front_path'] = front['path']
                selected += frames(front['path'], seg['start_frame'], seg['end_frame'], 8)
            # 각 행은 왼쪽부터 시간순이며 다른 카메라는 다음 행에 놓는다.
            canvas = Image.new('RGB', (1792, 190*(len(selected)//8)+35), '#121820'); draw = ImageDraw.Draw(canvas)
            draw.text((4, 5), f'{n:03d} {label["id"]} [{label["split"]}]',font=FONT,fill='white')
            for j,(idx,im) in enumerate(selected):
                im.thumbnail((224,160)); x=(j%8)*224; y=35+(j//8)*190
                canvas.paste(im,(x+(224-im.width)//2,y)); draw.text((x+3,y+163),f'f{idx} {idx/seg["fps"]:.2f}s',font=FONT,fill='white')
            canvas.save(ROOT/f'case_{n:03d}.jpg',quality=92)
            row['review_sheet']=str(ROOT/f'case_{n:03d}.jpg'); rows.append(row)
    for j in range(0,len(rows),4):
        ims=[Image.open(r['review_sheet']) for r in rows[j:j+4]]; out=Image.new('RGB',(1792,sum(im.height for im in ims)),'black'); y=0
        for im in ims:out.paste(im,(0,y));y+=im.height
        out.save(ROOT/f'panel_{j//4+1:02d}.jpg',quality=93)
    (ROOT/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    inventory = {}
    for ex in ['pushups','squats']:
        rs=[r for r in rows if r['exercise']==ex]
        inventory[ex]=dict(scored_drafts=len(rs),people=len(set(r['person'] for r in rs)),splits=dict(Counter(r['split'] for r in rs)),items={item:dict(Counter(str(r['criterion_levels'][i]) for r in rs)) for i,item in enumerate(rs[0]['criterion_ids'])})
    (ROOT/'inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2));print(json.dumps(inventory,ensure_ascii=False))

if __name__ == '__main__': main()
