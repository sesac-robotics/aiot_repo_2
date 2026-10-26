"""Review complete sources at 2 fps, preserving already frozen development roles."""
import json
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

B = Path(__file__).resolve().parents[2]
O = B/'output/data-readiness-20261024/dense-review'
O.mkdir(parents=True, exist_ok=True)
pool = json.loads((B/'output/return-failure-expansion-20261024/tail-review/intake.frozen.json').read_text())
rows = [r for r in pool if r['review_id'] in ['T014','T016','T023','T026']]
assert len(rows)==4 and all(r['split'] in ['train','validation'] for r in rows)
manifest = []
for r in rows:
    cap = cv2.VideoCapture(r['path'], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS,1])
    frames = list(range(0, r['frame_count'], max(1, round(r['fps']/2))))
    current = -1
    sheets = []
    for offset in range(0, len(frames), 32):
        im = Image.new('RGB', (8*220,4*210), 'white')
        draw = ImageDraw.Draw(im)
        for j, f in enumerate(frames[offset:offset+32]):
            while current < f:
                ok = cap.grab(); current += 1
                assert ok
            ok, a = cap.retrieve(); assert ok
            thumb = Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
            thumb.thumbnail((220,184))
            x, y = j%8*220, j//8*210
            im.paste(thumb,(x+(220-thumb.width)//2,y+24))
            draw.text((x+2,y+3),f"{r['review_id']} {f/r['fps']:.2f}s f{f}",fill='black')
        target = O/f"{r['review_id']}-full-{offset//32+1:02}.jpg"
        im.save(target, quality=94)
        sheets.append(str(target))
    cap.release()
    manifest.append(dict(**r, reviewed_frame_indices=frames, sheets=sheets, review_status='pending_visual_review', grade=None))
    print('PREPARED',r['review_id'],len(frames),len(sheets),flush=True)
(O/'intake.frozen.json').write_text(json.dumps(manifest,indent=2))
