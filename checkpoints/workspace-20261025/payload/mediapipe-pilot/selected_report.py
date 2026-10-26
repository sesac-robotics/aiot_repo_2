from pathlib import Path
import json,csv,re,statistics,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pilot import ROOT,JOINTS

out=ROOT/'outputs/selected-exercise';out.mkdir(exist_ok=True)
samples=json.loads((ROOT/'configs/samples.json').read_text());plan=json.loads((ROOT/'outputs/disk-plan-updated.json').read_text());est=plan['_estimate']
records=[];resources=[];packed=[]
for m in ['lite','full','heavy']:
    summaries=json.loads((out/'gpu'/m/'summary.json').read_text())
    log=(out/'gpu'/f'{m}.log').read_text()
    assert log.count('renderer: NVIDIA GeForce RTX 3050 Ti')==3
    rs=json.loads((out/'gpu'/m/'resources.json').read_text());gpu=[list(map(float,s['whole_gpu_util_memory'].split(','))) for s in rs if s['whole_gpu_util_memory']]
    resources.append({'model':m,'process_rss_peak_mib':max(s['process_rss_mb'] for s in rs),'system_cpu_mean_pct':statistics.mean(s['system_cpu_pct'] for s in rs),'system_cpu_peak_pct':max(s['system_cpu_pct'] for s in rs),'whole_gpu_peak_pct':max(x[0] for x in gpu),'whole_gpu_peak_memory_mib':max(x[1] for x in gpu),'available_ram_min_mib':min(s['system_available_mb'] for s in rs)})
    for rec in summaries:
        rec['delegate']='gpu';records.append(rec);folder=out/'gpu'/m/rec['dataset'];rows=list(csv.DictReader((folder/'angles.csv').open()));n=len(rows)
        assert rec['video_verified_frames']==rec['frames']==n
        normalized=np.full((n,33,5),np.nan,np.float32);world=np.full((n,33,3),np.nan,np.float32)
        for i,line in enumerate((folder/'landmarks.jsonl').open()):
            x=json.loads(line)
            if x['normalized_landmarks']:normalized[i]=np.asarray(x['normalized_landmarks'],np.float32);world[i]=np.asarray(x['world_landmarks'],np.float32)[:,:3]
        keys=[f'{j}_{dim}_deg' for j in JOINTS for dim in ['2d','3d']]
        angles=np.array([[float(r[k]) if r[k] else np.nan for k in keys] for r in rows],dtype=np.float32)
        assert np.all((angles[np.isfinite(angles)]>=0)&(angles[np.isfinite(angles)]<=180))
        timestamps=np.array([int(r['timestamp_ms']) for r in rows],dtype=np.int64)
        np.savez_compressed(folder/'features.npz',normalized_landmarks=normalized,world_xyz=world,angles=angles,timestamp_ms=timestamps,angle_names=np.array(keys))
        check=np.load(folder/'features.npz');assert np.array_equal(check['angles'],angles,equal_nan=True);assert check['world_xyz'].shape==(n,33,3)
        packed.append({'model':m,'dataset':rec['dataset'],'frames':n,'compressed_bytes':(folder/'features.npz').stat().st_size,'eight_3d_angles_simultaneously_valid':int(np.all(np.isfinite(angles[:,1::2]),axis=1).sum()),'valid_3d_angle_counts':{k:int(np.isfinite(angles[:,2*i+1]).sum()) for i,k in enumerate(JOINTS)}})
(out/'compact-feature-check.json').write_text(json.dumps(packed,indent=2));(out/'resource-summary.json').write_text(json.dumps(resources,indent=2))
fig,axes=plt.subplots(1,3,figsize=(14,4))
for ax,sample in zip(axes,samples):
 rr=[r for r in records if r['dataset']==sample['dataset']]
 ax.bar([r['model'] for r in rr],[r['pose_mean_ms'] for r in rr],color=['#4f9d90','#4086ac','#8065a4']);ax.set_title(sample['dataset'].replace('Pose-Estimation-Model-Comparison','Pose Comparison'));ax.set_ylabel('Mean inference ms / frame');ax.spines[['top','right']].set_visible(False)
fig.suptitle('Squat pilot | NVIDIA RTX 3050 Ti | 640 px max edge | 2026-10-16');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
for s in samples:
 fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True,sharey=True)
 for ax,j in zip(axes.flat,JOINTS):
  for m in ['lite','full','heavy']:
   rr=list(csv.DictReader((out/'gpu'/m/s['dataset']/'angles.csv').open()))
   ax.plot([float(r['timestamp_ms'])/1000 for r in rr],[float(r[j+'_3d_deg']) if r[j+'_3d_deg'] else np.nan for r in rr],label=m,lw=1)
  ax.set_title(j);ax.set_ylim(0,185);ax.set_xlabel('seconds');ax.grid(alpha=.2)
 axes[0,0].legend();fig.suptitle(s['dataset']+' | estimated 3D angles; missing data stays blank');fig.tight_layout();fig.savefig(out/(s['dataset']+'-angles.png'),dpi=130);plt.close(fig)

lines=['# MediaPipe 스쿼트 실행·관절 각도·자원 사용량 검증 보고서','','실행일: 2026-10-16 · 임시 실험 · NVIDIA GPU Lite → Full → Heavy 순차 실행','','## 1. 결론과 요청별 확인','','| 요청 | 실제 확인 결과 |','|---|---|',
'| 가장 가벼운·중간·무거운 모델 | Lite → Full → Heavy 순서로 실제 GPU 실행, 각각 3개 데이터셋의 스쿼트 영상 처리 |',
'| 관절 각도 | MediaPipe는 관절 좌표를 출력하며, 그 좌표에서 각도를 별도 계산해 CSV에 저장 |',
'| 여러 관절 | 팔꿈치·무릎·고관절·어깨의 좌우 8개를 같은 프레임에서 계산. 가림·미검출은 빈 값 |',
'| 자원 | 프로세스 RAM·CPU와 시스템 CPU·남은 RAM·NVIDIA 사용률·VRAM 기록 |',
'| train 입력과 표시 | REHAB26 제안 fold_0 train과 QEVD manifest train 영상 읽기·스켈레톤·각도 표시 성공. 세 번째 자료는 공식 분할이 없어 미리보기로 구분 |',
'| 보고서·결과물 | MP4·JPEG·CSV·JSONL·압축 NPZ·그래프·원본 라벨·실행 로그 생성 및 검사 |',
'', '이 결과는 사전 학습 MediaPipe를 이용한 입력·추론·각도 계산 검증이다. 새로운 교정 모델 학습, 오류 판별 정확도 평가, 전체 영상 전수 검수, 웹캠 실시간 실행을 완료했다는 뜻은 아니다.','',
'## 2. 범위 변경과 디스크 계획','','사용자 요청 전체 범위는 REHAB26의 3종 전부와 Pose Comparison의 4종 전부, QEVD의 스쿼트·일반 팔굽혀펴기·런지·점핑잭·팔 옆으로 들기·어깨 뒤로 젖히기 6종이다. QEVD의 마지막 두 종목은 정확히 대응하는 라벨을 찾지 못했다. 팔 돌리기·어깨 워밍업 등으로 임의 대체하지 않는다. 마지막 두 종목은 REHAB26 ex1·ex7에서 확인된다.',
'디스크 추정은 QEVD에서 일치하는 4개 정확한 라벨 squats/pushups/lunges/jumping jacks로 계산했다. lunges의 번갈아 수행·좌우·점프 등 별도 라벨은 포함하지 않았으며, 포함 범위를 넓히면 다시 계산해야 한다. 요청 6종을 모두 QEVD에서 확보했다고 표시하지 않는다.','',
'| 항목 | 예상 용량 | 근거 |','|---|---:|---|',
f"| 선택 범위 원본 영상 | {est['selected_source_bytes']/1e9:.2f} GB | 실제 파일 크기 합계. 이미 T9에 있어 추가 복사 불필요 |",
f"| 모델 1개 좌표·각도 배열 | {est['binary_bytes_per_model']/1e9:.2f} GB | 약 {est['total_frames']:,.0f}프레임 × 프레임당 1,128바이트 |",
f"| 모델 3개 좌표·각도 배열 | {est['binary_three_models']/1e9:.2f} GB | float32 정규화 좌표/신뢰도·world XYZ·16개 각도·int64 시각. NPZ 압축 전 |",
'| 추가 디스크 권장 확보 | 15~20 GB | 배열 3개 모델·목록·로그·대표 결과 영상·여유. 전체 결과 영상 저장 제외 |',
f"| 현재 JSONL 방식, 모델 3개 | 약 {est['json_three_models']/1e9:.1f} GB | 짧은 QEVD 실측 약 7,007바이트/프레임 적용. 각도 CSV·영상·로그는 별도 |",
f"| 내부 디스크 남은 용량 | {est['free_internal']/2**30:.1f} GiB | 이번 확인 시점 |",
f"| T9 남은 용량 | {est['free_T9']/2**40:.2f} TiB | 이번 확인 시점 |",
'', 'REHAB26 45개와 Pose Comparison 4개는 전 파일 메타데이터를 읽었다. QEVD는 종목당 80개 무작위 표본(시드 37)으로 프레임 수를 추정했고, 영상 수·파일 크기는 선택 목록 전체를 집계했다. 전체를 디코딩한 수치가 아니므로 영상 손상이나 가변 프레임레이트 등을 모두 검증한 것은 아니다. 이번 샘플은 NPZ를 실제 생성·재로드 확인했지만, 전체 압축률은 보장하지 않는다. 프레임 JPEG 전체 추출·원본 복사·모든 영상 재인코딩은 계획에 포함하지 않는다.',
'', 'GB는 10억 바이트, GiB는 2의 30승 바이트다. 원본 압축파일·원 제공 좌표 등 T9에 이미 있는 전체 자료의 총량과 이번 추가 출력 용량은 다르다. 최초 스쿼트·브리지·무릎 팔굽혀펴기·점핑잭 가정의 disk-plan.json은 이전 계산이며, 최신 범위는 disk-plan-updated.json이다.','',
'## 3. 환경과 실행 조건','','- 장치: Intel i7-12700H, 14코어/20스레드, RAM 약 16GB, NVIDIA RTX 3050 Ti Laptop 4GB.',
'- 공통 가상환경: /home/jayden/SeSACPAI-Mid/aiot_repo_2/.venv. 패키지 버전은 requirements-lock.txt, pip check 통과.',
'- 모델: Google 공식 Pose Landmarker Lite/Full/Heavy float16 version 1. 다운로드 URL·SHA-256은 models/sources.json.',
'- VIDEO 모드, 최대 1명, 검출·존재·추적 임계값 0.5, 원본 비율 유지 최대 변 640픽셀, 프레임 건너뛰기 없음.',
'- 각 모델 별도 프로세스, 영상별 새 추적기, 첫 5프레임은 추론 시간 통계에서 제외. 모델 로딩 시간은 metrics.json에 별도 기록.',
'- GPU는 OpenGL ES delegate. PRIME/EGL 실행 환경으로 RTX를 선택했으며 로그에 3개 영상 모두 NVIDIA renderer가 있다. 기본 GPU 선택은 Intel 내장 GPU였으므로 GPU 옵션만 켜면 RTX가 된다고 가정하지 않는다.',
'- 현재 로그에는 NORM_RECT/IMAGE_DIMENSIONS 및 feedback tensor 관련 경고가 남아 있다. 실행 실패는 없었지만 좌표 정밀도는 정답과 별도 대조해야 한다.',
'', '## 4. 선택한 영상과 라벨','','| 자료 | 영상 | 분할 | 실제 처리 | 라벨 |','|---|---|---|---|---|']
for s in samples:
 rec=next(x for x in records if x['model']=='lite' and x['dataset']==s['dataset']);duration=rec['frames']/s['video_metadata']['fps']
 lines.append(f"| {s['dataset']} | {s['video_id']} | {s['split']} | {rec['frames']}프레임 / {duration:.2f}초 | {s['label_kind']} |")
lines+=['','REHAB26은 첫 900프레임(30초)만 처리했으며 긴 원본 전체 처리는 아니다. QEVD와 Pose Comparison은 선택 클립을 끝까지 처리했다. REHAB26의 분할은 공식 배포 train이 아니라 기존 로컬 제안 fold_0이다. Pose Comparison은 train/validation/test를 임의 배정하지 않았다. 라벨 세부값과 전체 영상 경로는 각 sample-and-labels.json에서 확인할 수 있다.','',
'QEVD 선택 영상 라벨: '+', '.join(next(s['labels'] for s in samples if s['dataset']=='QEVD')),
'', '## 5. 실측 성능','','| 자료 | 모델 | 평균 추론 ms | p95 ms | 읽기·크기변경·추론·각도 평균 ms | 결과 저장 포함 처리 FPS | 검출 프레임 |','|---|---|---:|---:|---:|---:|---:|']
for r in records:lines.append(f"| {r['dataset']} | {r['model']} | {r['pose_mean_ms']:.2f} | {r['pose_p95_ms']:.2f} | {r['pipeline_mean_ms_decode_resize_pose_angles']:.2f} | {r['preview_throughput_fps_including_csv_json_drawing_encoding']:.1f} | {r['detected_frames']}/{r['frames']} |")
lines+=['','![실측 추론 시간](../outputs/selected-exercise/comparison.png)','','추론 시간은 detect_for_video 호출만 측정한다. pipeline에는 원본 읽기·디코딩·크기/색 변환·추론·각도 계산이 포함되며, 결과 저장 포함 FPS에는 CSV/JSON 기록·그리기·초기 MP4 인코딩까지 포함한다. 후처리 H.264 변환·영상 재검사는 FPS 범위 밖이다. 검출률은 올바른 운동 판정률·관절 위치 정확도가 아니다. 실시간 웹캠 FPS로도 단정하지 않는다. 한 차례 실행이므로 통계적 성능 우위를 확정하지 않는다.','',
'## 6. 자원 사용량','','| 모델 | 프로세스 최대 RAM MiB | 시스템 CPU 평균 % | 시스템 CPU 최대 % | NVIDIA 전체 최대 사용률 % | NVIDIA 전체 최대 VRAM MiB | 최소 여유 RAM MiB |','|---|---:|---:|---:|---:|---:|---:|']
for r in resources:lines.append(f"| {r['model']} | {r['process_rss_peak_mib']:.0f} | {r['system_cpu_mean_pct']:.1f} | {r['system_cpu_peak_pct']:.1f} | {r['whole_gpu_peak_pct']:.0f} | {r['whole_gpu_peak_memory_mib']:.0f} | {r['available_ram_min_mib']:.0f} |")
lines+=['','약 0.5초 간격 표본이므로 매우 짧은 최대치는 놓칠 수 있다. GPU 사용률·VRAM과 시스템 CPU는 다른 앱을 포함한다. MediaPipe 단독 GPU 메모리라고 주장하지 않는다. 프로세스 CPU는 각 metrics.json에 100%=논리 코어 1개 기준으로 따로 저장했다. 디코딩·렌더링·결과 저장은 GPU를 선택해도 CPU를 사용한다.','',
'## 7. 관절 각도 확인','','세 점 A-B-C의 벡터 내적에서 B의 0~180도 사이각을 계산했다. 2D는 가로·세로 픽셀 좌표를 사용해 비율 왜곡을 피하고, 3D는 world XYZ 추정값을 사용했다. 2D/3D 각각 8개 각도다. 90도·180도·영벡터 사례로 수식을 확인했고 CSV의 모든 유효값이 0~180도인지 검사했다. 임상적 관절 가동범위나 실제 회전각의 정확도를 검증한 것은 아니다.','',
'| 자료 | 모델 | 8개 3D 각도가 같은 프레임에서 모두 유효 | 처리 프레임 |','|---|---|---:|---:|']
for r in packed:lines.append(f"| {r['dataset']} | {r['model']} | {r['eight_3d_angles_simultaneously_valid']} | {r['frames']} |")
lines+=['','관절 3점 모두 visibility/presence ≥ 0.5이고 영상 안에 있을 때만 해당 각도를 계산한다. 미검출·가림·영벡터는 CSV 빈칸과 배열 NaN으로 저장한다. 다른 쪽 관절이나 이전 프레임 값으로 채우지 않는다. Pose Comparison 측면 영상에서는 반대쪽 관절이 가려져 4개 각도만 유효한 경우가 있다. 여러 각도를 계산할 수 있다는 것과 모든 각도를 매 순간 관측할 수 있다는 것은 다르다.','',
'## 8. 결과물과 다음 단계','','- 결과 선택 화면: ../selected-exercise.html', '- 각도·좌표·원본 라벨·자원 기록: ../outputs/selected-exercise/gpu/<모델>/<자료> 및 모델 resources.json', '- 영상: 9개 MP4 모두 재열기·끝까지 디코딩하여 입력 처리 프레임 수와 일치 확인.', '- 압축 배열: 9개 features.npz를 실제 생성 후 재로드·배열 값/형상 대조.', '- T9 원본과 정식 aiot_repo_2는 이번 실험에서 변경하지 않았다.',
'', '다음은 출처별 스쿼트의 여러 수행자·촬영 방향으로 확대하고 오류별 라벨 분포·분할 누출·관절 좌표/각도 정답을 확인하는 단계다. 교정 모델 학습 시간은 모델 구조·배치·에포크를 정한 뒤 소량 학습으로 실측해야 한다. 강사 승인 전 정식 폴더 구조나 원격 저장소에 이 코드를 반영하지 않는다.','',
'## 공식 참고','','- https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker','- https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python','']
reports=ROOT/'reports';reports.mkdir(exist_ok=True)
schedule_path=reports/'estimated-schedule.md'
if schedule_path.exists():
    at=lines.index('## 공식 참고')
    lines[at:at+1]=[schedule_path.read_text(), '', '## 10. 공식 참고']
(reports/'2026-10-16-mediapipe-squat-pilot.md').write_text('\n'.join(lines))
template=(ROOT/'index.html').read_text();template=template.replace('<option value="cpu">CPU</option>','');template=template.replace("'outputs/'+d+'/'+m+'/'+n+'/'", "'outputs/selected-exercise/'+d+'/'+m+'/'+n+'/'")
template=template.replace('src="outputs/comparison.png"','src="outputs/selected-exercise/comparison.png"').replace("'outputs/'+n+'-angles.png'","'outputs/selected-exercise/'+n+'-angles.png'")
template=template.replace('href="RESULTS.md"','href="reports/2026-10-16-mediapipe-squat-pilot.md"').replace('T9 원본 3개 · Lite / Full / Heavy · CPU / RTX 3050 Ti','데이터셋별 스쿼트 1개 · Lite / Full / Heavy · RTX 3050 Ti')
data=json.dumps({'records':records,'samples':samples,'resources':resources},ensure_ascii=False).replace('</','<\\/')
template=re.sub(r'const DATA=.*?;const device=',lambda m:'const DATA='+data+';const device=',template,flags=re.S)
(ROOT/'pose-only-reference.html').write_text(template)
print(json.dumps({'runs':len(records),'frames':sum(r['frames'] for r in records),'resources':resources,'disk_GB':est['binary_three_models']/1e9},ensure_ascii=False,indent=2))
