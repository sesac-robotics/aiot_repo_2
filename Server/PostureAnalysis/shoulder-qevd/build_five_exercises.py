"""Build shared label definitions and QEVD-format annotations for five exercises.
Run from VS Code with Python 3. Videos remain at their original locations.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import html
import json
import re
from urllib.parse import urlencode

BASE = Path(__file__).resolve().parent
OUT = BASE / 'five_exercises'
NAMES = {'shoulder_press': '숄더 프레스', 'squats': '스쿼트', 'pushups': '푸시업', 'elbow_plank': '엘보 플랭크', 'ytw': 'YTW'}

def read(path):
    return json.loads(path.read_text())

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def main():
    definitions = []
    for item in read(BASE.parent / 'qevd-selected/derived/manifests/feedback_label_mapping.json'):
        slug = re.sub(r'[^A-Z0-9]+', '_', item['name_en'].upper()).strip('_')
        code = item['exercise_group'].upper() + '_' + slug + '_' + hashlib.sha256(item['label'].encode()).hexdigest()[:8].upper()
        definitions.append(dict(code=code, exercise=item['exercise_group'], label=item['label'], name_en=item['name_en'], descriptions_en=item['original_descriptions'], feedback_en=item['feedback_en'], category=item['category'], label_origin='source_dataset', feedback_status='draft', expert_verified=False))
    for item in read(BASE / 'manifests/label_definitions.json'):
        definitions.append(dict(code='SHOULDER_PRESS_' + item['code'], exercise='shoulder_press', label=item['label'], name_en=item['name_en'], descriptions_en=[item['labels_descriptive']], feedback_en=item['feedback_en'], category=item['annotation_type'], label_origin='source_binary_or_visual_draft', feedback_status='draft', expert_verified=False))
    lookup = {d['label']: d for d in definitions}
    labels, feedbacks, workers, records = [], [], [], []
    counts = Counter()
    for dataset, root in [('qevd', BASE.parent / 'qevd-selected'), ('shoulder', BASE)]:
        with (root / 'manifests/short_clips.csv').open() as stream:
            manifest = {r['video_id']: r for r in csv.DictReader(stream)}
        feedback_path = root / ('derived/coach/feedbacks_short_clips.json' if dataset == 'qevd' else 'raw/coach/feedbacks_short_clips.json')
        cues = {Path(r['video_path']).stem: r for r in read(feedback_path)}
        for row in read(root / 'raw/coach/fine_grained_labels.json'):
            vid = Path(row['video_path']).stem
            meta = manifest[vid]
            path = (root / meta['video_path']).resolve()
            assert path.is_file(), path
            defs = [lookup[label] for label in row['labels']]
            groups = {d['exercise'] for d in defs}
            assert len(groups) == 1
            exercise = groups.pop()
            counts[exercise] += 1
            labels.append(row)
            feedbacks.append(cues[vid])
            workers.append({'video_path': row['video_path'], 'worker_id': int(meta['worker_id'])})
            records.append(dict(video_id=vid, dataset=dataset, video_path=str(path), split=row['split'], worker_id=int(meta['worker_id']), exercise=exercise, label_codes=[d['code'] for d in defs], labels=row['labels'], labels_descriptive=row['labels_descriptive'], feedbacks=cues[vid]['feedbacks'], label_status='source_dataset_annotation_unassessed' if dataset == 'qevd' else 'draft', feedback_status='draft', expert_verified=False, training_ready=False, segments=None, landmarks=None, review_required=['visual_verification', 'temporal_annotation', 'pose_extraction'] + (['ytw_body_variant_and_intended_order'] if exercise == 'ytw' else [])))
    assert len(records) == len({r['video_id'] for r in records})
    assert len(lookup) == len(definitions) == 70
    assert set(counts) == set(NAMES)
    subjects = {split: {r['worker_id'] for r in records if r['split'] == split} for split in ('train', 'test')}
    assert not subjects['train'] & subjects['test']
    save(OUT / 'manifests/label_definitions.json', definitions)
    save(OUT / 'manifests/annotations.json', records)
    save(OUT / 'raw/coach/fine_grained_labels.json', labels)
    save(OUT / 'raw/coach/feedbacks_short_clips.json', feedbacks)
    save(OUT / 'raw/annotations/fine_grained_labels_with_worker_ids.json', workers)
    with (OUT / 'manifests/short_clips.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['video_id', 'video_path', 'split', 'worker_id', 'labels', 'labels_descriptive'])
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], list) else row[k] for k in writer.fieldnames})
    report = dict(passed=True, videos=len(records), label_types=len(definitions), videos_by_exercise=dict(counts), label_types_by_exercise=dict(Counter(d['exercise'] for d in definitions)), unique_ids=True, all_video_paths_exist=True, worker_split_overlap=0, verified_new_annotations=0, note='Clip-level source labels plus draft feedback; not frame-level or model predictions.')
    save(OUT / 'checks/validation_report.json', report)
    sections = []
    esc = html.escape
    for exercise, name in NAMES.items():
        cards = []
        for d in definitions:
            if d['exercise'] != exercise:
                continue
            candidates = [r for r in records if d['code'] in r['label_codes']]
            sample = candidates[0]
            url = '/?' + urlencode(dict(dataset=sample['dataset'], exercise=exercise, q=sample['video_id'], autoplay=1, demo=1))
            cards.append(f'<article><code>{esc(d["code"])}</code><h3>{esc(d["name_en"])}</h3><small>{esc(d["category"])} · {len(candidates)} videos</small><p>{esc(d["descriptions_en"][0])}</p><p><b>Feedback:</b> {esc(d["feedback_en"])}</p><a href="{esc(url)}">영상 + 교정 문구 보기 →</a></article>')
        sections.append(f'<section id="{exercise}"><h2>{name} · {counts[exercise]:,} videos · {len(cards)} labels</h2><div class="grid">{"".join(cards)}</div></section>')
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>5종목 자세 라벨링</title><style>body{font:15px system-ui;background:#f4f7fb;color:#20334e;margin:0}main{max-width:1200px;margin:auto;padding:28px}nav{display:flex;gap:20px;flex-wrap:wrap}a{color:#245bc1}.notice{background:#fff0ce;padding:18px;border-radius:10px;line-height:1.8}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}article{background:white;padding:20px;border:1px solid #dce3ee;border-radius:12px}code{overflow-wrap:anywhere;font-size:12px}p{line-height:1.7}section{margin-top:36px}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style><main><h1>5종목 자세 오류 라벨링</h1><p>18,752개 영상 · 공통 라벨 70종 · 영어 설명 및 교정 문구</p><nav>' + ''.join(f'<a href="#{k}">{v}</a>' for k,v in NAMES.items()) + '</nav><p class="notice">QEVD 원본 라벨 63종 + 어깨 초안 7종입니다. 정상 후보·동작 특성·오류 후보를 구분했습니다. 교정 문구는 검수 전 초안이며, 모든 영상을 새로 수동 검수한 결과가 아닙니다. YTW는 몸통 변형과 목표 순서를 먼저 확인해야 합니다. 시간 구간과 MediaPipe 좌표는 아직 없습니다.</p>' + ''.join(sections) + '</main></html>'
    (BASE / 'test_app/web/five_exercises.html').write_text(page)
    (OUT / 'README.md').write_text('''# 5종목 공통 라벨링 데이터

`../build_five_exercises.py`를 실행해 재생성합니다. `../run_five_exercises.py`는 영상 확인 화면을 엽니다.

- `manifests/label_definitions.json`: 공통 코드, 운동, 원본 라벨, 영어 설명, 교정 문구, 유형 및 출처.
- `manifests/annotations.json`: 영상별 다중 라벨 코드, 실제 경로, split, worker, 검토 상태.
- `raw/coach/`: QEVD와 같은 JSON 필드 구성. 원본 라벨 문구와 split을 유지합니다.
- `manifests/short_clips.csv`: QEVD와 같은 열 구성. 영상 복제 없이 절대 경로로 기존 영상을 참조합니다.
- `checks/validation_report.json`: 경로/코드/ID/피험자 분할 검사.

전체 18,752개 중 QEVD 18,652개는 기존 원본 라벨을 재사용하고, 어깨 100개는 기존 초안을 사용합니다. 라벨 70종 모두가 오류는 아닙니다. 정상 후보, 자세 특성, 동작 속도·순서, 가시성, 모호하거나 충돌하는 라벨도 별도 category로 보존합니다.

교정 문구는 초안이며 새 전문가 검수는 없습니다. training_ready=false는 이번 통합본의 검수 기준이며 QEVD 원본 자체가 학습 불가능하다는 의미는 아닙니다. YTW는 서 있는 몸통/숙인 몸통 변형을 아직 영상별 판정하지 않았습니다. segments=null, landmarks=null이며 오류 발생 프레임을 뜻하지 않습니다. 실제 학습 전 목표 변형 확인, 수동 검수, 포즈 추출이 필요합니다. 어깨의 미확정 40개와 보류 10개는 제외했습니다.
''')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
