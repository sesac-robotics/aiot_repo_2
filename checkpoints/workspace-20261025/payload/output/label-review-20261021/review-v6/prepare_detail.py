"""판단이 애매한 푸시업 구간을 확대하여 재검토할 근거를 만든다."""
import json
from pathlib import Path
from PIL import Image, ImageDraw
from prepare_review import frames, FONT
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'expanded'
# 직접 확인한 경계와, 확인이 필요한 근접 구간이다. 등급을 자동 부여하지 않는다.
REQUESTS={3:(10,28),4:(0,45),6:(40,59),13:(82,95),17:(31,64),27:(5,35),28:(33,50),42:(15,60),47:(95,115),10:(25,55),16:(40,70),32:(35,55),43:(15,35),44:(15,35),48:(15,35)}


def main():
    """확대판과 모든 원본 프레임의 시간순 접촉판을 저장한다.

    Returns:
        None: detail_*.jpg와 dense_*.jpg를 저장한다.
    Raises:
        RuntimeError: 프레임 읽기 실패 시 발생한다.
    """
    for n,(a,b) in REQUESTS.items():
        meta=json.loads((OLD/f'{n:03d}.json').read_text())
        images=frames(meta['path'],a,b,9);out=Image.new('RGB',(1536,3*310),'#121820');d=ImageDraw.Draw(out)
        for j,(idx,im) in enumerate(images):
            im.thumbnail((512,285));x=j%3*512;y=j//3*310;out.paste(im,(x,y));d.text((x+5,y+287),f'{n} f{idx} {idx/15:.3f}s',font=FONT,fill='white')
        out.save(ROOT/f'detail_{n:03d}.jpg',quality=95)
        images=frames(meta['path'],a,b,b-a+1);out=Image.new('RGB',(1792,((len(images)+7)//8)*150),'#121820');d=ImageDraw.Draw(out)
        for j,(idx,im) in enumerate(images):
            im.thumbnail((224,126));x=j%8*224;y=j//8*150;out.paste(im,(x,y));d.text((x+2,y+128),f'f{idx} {idx/15:.3f}s',font=FONT,fill='white')
        out.save(ROOT/f'dense_{n:03d}.jpg',quality=94)
    print('확대 및 연속 원본 프레임 근거 15건 생성')

if __name__=='__main__':main()
