from pathlib import Path
import csv,json,html,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pilot import ROOT, JOINTS

out=ROOT/'outputs'
records=[];resource_rows=[]
for delegate in ['cpu','gpu']:
    for model in ['lite','full','heavy']:
        records.extend(json.loads((out/delegate/model/'summary.json').read_text()))
        for r in records[-3:]:r['delegate']=delegate
        samples=json.loads((out/delegate/model/'resources.json').read_text())
        assert samples
        gpu=[list(map(float,s['whole_gpu_util_memory'].split(','))) for s in samples if s['whole_gpu_util_memory']]
        resource_rows.append({'delegate':delegate,'model':model,'peak_rss_mb':max(s['process_rss_mb'] for s in samples),'mean_system_cpu_pct':statistics.mean(s['system_cpu_pct'] for s in samples),'peak_system_cpu_pct':max(s['system_cpu_pct'] for s in samples),'min_available_ram_mb':min(s['system_available_mb'] for s in samples),'whole_gpu_util_pct_peak':max(x[0] for x in gpu),'whole_gpu_memory_mb_peak':max(x[1] for x in gpu),'sample_count':len(samples)})
with (out/'comparison.csv').open('w') as f:
    fields=['delegate','model','dataset','frames','detected_rate','pose_mean_ms','pose_p95_ms','pipeline_mean_ms_decode_resize_pose_angles','preview_throughput_fps_including_csv_json_drawing_encoding','process_cpu_pct_one_core_mean']
    w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(records)
(out/'resource-summary.json').write_text(json.dumps(resource_rows,indent=2))
samples=json.loads((ROOT/'configs/samples.json').read_text())
inventory=json.loads((out/'dataset-inventory.json').read_text())
aggregate=[]
for d in ['cpu','gpu']:
 for m in ['lite','full','heavy']:
    rr=[r for r in records if r['delegate']==d and r['model']==m]
    weights=[r['frames']-5 for r in rr]
    mean=float(np.average([r['pose_mean_ms'] for r in rr],weights=weights))
    resource=next(x for x in resource_rows if x['delegate']==d and x['model']==m)
    aggregate.append({'delegate':d,'model':m,'mean_ms':mean,'equivalent_fps':1000/mean,**resource})
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for i,d in enumerate(['cpu','gpu']):
 rows=[r for r in aggregate if r['delegate']==d]
 axes[0].bar(np.arange(3)+(i-.5)*.35,[r['mean_ms'] for r in rows],width=.35,label=d.upper(),color=['#627d98','#0f8b8d'][i])
 axes[1].bar(np.arange(3)+(i-.5)*.35,[r['peak_rss_mb'] for r in rows],width=.35,label=d.upper(),color=['#627d98','#0f8b8d'][i])
for a in axes:a.set_xticks(range(3),['Lite','Full','Heavy']);a.legend();a.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Pose inference ms / frame (lower is faster)');axes[1].set_ylabel('Process peak RSS MiB (0.5s samples)')
fig.suptitle('MediaPipe pilot | i7-12700H / RTX 3050 Ti | 3 squat clips, CPU vs GPU')
fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
for sample in samples:
 name=sample['dataset'];fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True,sharey=True)
 for ax,joint in zip(axes.flat,JOINTS):
    for model in ['lite','full','heavy']:
        rows=list(csv.DictReader((out/'gpu'/model/name/'angles.csv').open()))
        ax.plot([float(r['timestamp_ms'])/1000 for r in rows],[float(r[joint+'_3d_deg']) if r[joint+'_3d_deg'] else np.nan for r in rows],label=model,lw=1)
    ax.set_title(joint);ax.set_ylim(0,185);ax.set_xlabel('seconds');ax.grid(alpha=.2)
 axes[0,0].legend();fig.suptitle(name+' | estimated 3D included angles, missing joints left blank')
 fig.tight_layout();fig.savefig(out/(name+'-angles.png'),dpi=130);plt.close(fig)

lines=['# MediaPipe 임시 실행 검증 결과','', '실행일: 2026-10-16 · 정식 팀 구조 승인 전 별도 실험','',
'## 완료 범위','',
'- T9 원본에서 데이터셋별 스쿼트 영상 1개씩, 총 3개를 읽었다. 같은 입력에 Lite → Full → Heavy 순서로 CPU 1회, NVIDIA GPU 1회를 실행했다.',
'- 모델/장치별로 별도 프로세스를 사용했다. 영상별 새 추적기를 생성했고, 각 영상 최대 180프레임(약 4~6초)만 처리했다. 전체 데이터셋 학습·전수 영상 검사는 수행하지 않았다.',
'- 팔꿈치·무릎·고관절·어깨 좌우 8개 각도의 2D/추정 3D 값을 CSV로 저장했다. 원본 관절 33개는 JSONL에 저장했다.',
'- 2D는 영상 가로·세로 비율을 반영한 픽셀 좌표, 3D는 MediaPipe world 좌표를 사용했다. 세 점이 이루는 0~180도 사이각이며 임상적 관절 운동범위·정확한 회전각으로 검증된 값은 아니다.',
'- 세 점 모두 visibility/presence ≥ 0.5이고 화면 안에 있을 때만 각도를 계산한다. 미검출·가림·영벡터는 빈 값으로 남긴다. 한쪽이 가려지면 반대쪽으로 값을 복사하지 않는다.',
'- 90도·180도·영벡터 예제로 각도 계산 함수를 확인했다. 실제 각도 정답과의 오차는 아직 측정하지 않았다.',
'- 영상에 스켈레톤·8개 각도·추론 시간을 표시했고, 생성한 18개 MP4를 다시 열어 모든 출력 프레임을 디코딩 확인했다. 원본 라벨은 각 결과의 sample-and-labels.json에 함께 보존했다.','',
'## 환경과 GPU','',
'i7-12700H(14코어/20스레드), RAM 약 16GB, RTX 3050 Ti Laptop 4GB. `../aiot_repo_2/.venv` 공통 환경을 사용하며 패키지 버전은 requirements-lock.txt, 장치 정보는 outputs/environment.json에 기록했다.',
'GPU 기본 선택은 Intel Iris Xe였다. NVIDIA PRIME/EGL 환경을 지정해 RTX 3050 Ti를 선택했다. outputs/gpu/*.log에 NVIDIA renderer와 실제 추론 결과가 있다. 이 GPU 경로는 MediaPipe의 OpenGL ES delegate이며 PyTorch CUDA 실행과 다르다.',
'자원 샘플은 약 0.5초마다 수집했다. 프로세스 RSS는 이 실행의 메모리, CPU 100%는 논리 코어 1개 사용 기준이다(20개 전체 대비는 20으로 나눈 값). NVIDIA 사용률·VRAM은 데스크톱 등 다른 프로세스까지 포함한 장치 전체 값이므로 모델 단독 사용량으로 해석하지 않는다.','',
'## 모델 비교','',
'평균 추론 시간은 영상별 첫 5프레임을 제외하고 프레임 수로 가중했다. 추론 시간은 detect_for_video 호출만 포함한다. FPS 환산은 실제 화면 FPS가 아니다.','',
'| 장치 | 모델 | 평균 추론 ms | 추론 FPS 환산 | 최대 프로세스 RAM MiB | 장치 전체 최대 VRAM MiB |',
'|---|---|---:|---:|---:|---:|']
for r in aggregate:lines.append(f"| {r['delegate'].upper()} | {r['model']} | {r['mean_ms']:.2f} | {r['equivalent_fps']:.1f} | {r['peak_rss_mb']:.0f} | {r['whole_gpu_memory_mb_peak']:.0f} |")
lines+=['','![모델 비교](outputs/comparison.png)','','## 데이터 입력·분할 확인','','| 자료 | 선택 영상 | 분할 | 처리 프레임 |','|---|---|---|---:|']
for s in samples:
 lines.append(f"| {s['dataset']} | {s['video_id']} | {s['split']} | {min(180,s['video_metadata']['frame_count'])} |")
lines+=['',f"REHAB26 영상 목록 {inventory['REHAB26']['videos']}개, fold_0 분할 수: {inventory['REHAB26']['split_counts_fold_0']}. fold_0은 로컬 제안이며 공식 train 정답이 아니다. 수행자 단위 분리를 유지한다.",
f"QEVD short clip 목록 {inventory['QEVD']['short_clip_manifest_rows']}행. 분할 수: {inventory['QEVD']['official_split_counts']}. 원본 manifest의 train 라벨을 읽었으며 사람 단위 누출 검증을 완료했다는 뜻은 아니다.",
'Pose Comparison은 원본 영상 4개이며 공식 train/val/test가 배정되지 않았다. 이번에는 squat.mp4를 미리보기만 했다. 다른 모델 출력 CSV를 사람 정답으로 간주하거나 같은 원본 영상의 결과를 여러 분할에 나누지 않는다.',
'REHAB26 라벨은 반복 구간·점수·오류 코드, QEVD 라벨은 클립 단위 속성이다. 원본 라벨에서 각도 정답이나 오류 판별 정확도가 바로 제공된다고 가정하지 않는다. 이 실험에서는 오류 분류기를 학습하지 않았다.','',
'## 다음 단계','',
'1. 결과 미리보기에서 관절 누락·촬영 방향을 확인하고 더 긴 영상·다른 운동으로 확대한다.',
'2. 공식/제안 분할을 확정하고 수행자·원본 영상 누출을 검사한다.',
'3. 같은 조건으로 반복 측정하고 GPU/CPU 비교를 확대한다. 모델 크기가 항상 더 높은 품질·속도를 보장하지 않는다.',
'4. 강사 승인 후 공통 데이터 어댑터·포즈 출력 계약에 맞춰 정식 저장소로 필요한 코드만 옮긴다.','',
'## 공식 참고','',
'- [Pose Landmarker 모델·출력](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)',
'- [Python 실행 안내](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)',
'- 모델 다운로드 주소와 SHA-256: models/sources.json','']
(ROOT/'RESULTS.md').write_text('\n'.join(lines))
payload={'records':records,'samples':samples,'resources':resource_rows}
template='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>MediaPipe 임시 실험 결과</title><style>body{font-family:system-ui;background:#101820;color:#ecf0f3;max-width:1200px;margin:32px auto;padding:20px}h1{font-size:28px}select{padding:12px;margin:8px;background:#203341;color:white;border:1px solid #62818d;border-radius:8px}video{width:100%;max-height:650px;background:black;border-radius:12px}section{background:#192933;padding:20px;border-radius:12px;margin:20px 0}a{color:#6fd2d4}img{width:100%}pre{white-space:pre-wrap;color:#c1d1db}.metrics{font-size:19px;color:#76e0bc}</style><h1>MediaPipe · 임시 실행 검증</h1><p>T9 원본 3개 · Lite / Full / Heavy · CPU / RTX 3050 Ti</p><p>정식 구조 승인 전 실험입니다. 각도는 추정값이며, 자세 교정 정확도 검증·전체 학습 결과가 아닙니다.</p><section><label>장치 <select id="device"><option value="gpu">RTX 3050 Ti GPU</option><option value="cpu">CPU</option></select></label><label>모델 <select id="model"><option>lite</option><option>full</option><option>heavy</option></select></label><label>자료 <select id="dataset"></select></label><p class="metrics" id="metrics"></p><video id="video" controls playsinline></video><pre id="details"></pre><p><a id="csv">관절 각도 CSV</a> · <a id="points">33관절 좌표 JSONL</a></p></section><section><h2>모델 비교</h2><img src="outputs/comparison.png"><p>추론 FPS 환산은 화면 재생 FPS와 다릅니다. GPU VRAM은 장치 전체 사용량입니다.</p></section><section><h2>여러 관절 각도 변화 · GPU</h2><img id="angles"></section><p><a href="RESULTS.md">상세 결과 문서</a></p><script>const DATA=__DATA__;const device=document.querySelector('#device'),model=document.querySelector('#model'),dataset=document.querySelector('#dataset');for(const s of DATA.samples){const o=document.createElement('option');o.value=s.dataset;o.textContent=s.dataset;dataset.appendChild(o)}function update(){const d=device.value,m=model.value,n=dataset.value,r=DATA.records.find(x=>x.delegate===d&&x.model===m&&x.dataset===n),s=DATA.samples.find(x=>x.dataset===n);const base='outputs/'+d+'/'+m+'/'+n+'/';document.querySelector('#video').src=base+'preview.mp4';document.querySelector('#metrics').textContent='추론 평균 '+r.pose_mean_ms.toFixed(2)+' ms · p95 '+r.pose_p95_ms.toFixed(2)+' ms · 관절 검출 '+r.detected_frames+'/'+r.frames+' 프레임';document.querySelector('#details').textContent='영상: '+s.video+'\\n분할: '+s.split+'\\n라벨: '+s.label_kind+'\\n관절마다 관측 가능성이 다르며 가림·미검출은 unavailable로 표시합니다.';document.querySelector('#csv').href=base+'angles.csv';document.querySelector('#points').href=base+'landmarks.jsonl';document.querySelector('#angles').src='outputs/'+n+'-angles.png'}for(const el of [device,model,dataset])el.onchange=update;update();</script></html>'''
(ROOT/'index.html').write_text(template.replace('__DATA__',json.dumps(payload,ensure_ascii=False).replace('</','<\\/')))
print(json.dumps({'runs':len(records),'aggregate':aggregate},ensure_ascii=False,indent=2))
