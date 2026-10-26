"""해병대 시범 영상의 정상·오류 후보 구간을 추출한다. 점수는 생성하지 않는다."""
import json
from prepare_details import ROOT,sheet
manifest=[]
for n,(start,end) in enumerate([(52,55),(57,60),(83,86),(92,95),(98,101),(104,107),(110,113),(116,119)],1):
    r=sheet(ROOT/'sources/marine_plank_720.mp4',start,end,ROOT/f'marine_{n:02d}.jpg');r.update(video_path=str(ROOT/'sources/marine_plank_720.mp4'),review_number=n);manifest.append(r)
(ROOT/'marine_segments.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
