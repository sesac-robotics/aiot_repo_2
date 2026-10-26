"""야간 실험을 점수 초안→점수 모델→오류 라벨 정리→오류 모델 순서로 실행한다.

중간 결과는 단계별로 보존한다. 외부 메시지 전송·기존 웹 교체·전문가 승인 처리는 하지 않는다.
"""
import collections
import fcntl
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import threading
from common import BASE, OUT, save, progress


def notify(title,text,failed=False):
    """Ubuntu 세션에 완료 또는 실패 알림을 보낸다. 로그에도 전송 결과를 남긴다."""
    env=os.environ.copy();env.setdefault('DBUS_SESSION_BUS_ADDRESS',f'unix:path=/run/user/{os.getuid()}/bus')
    result=subprocess.run(['notify-send','--app-name=야간 자세 평가 실험','--expire-time=0','--urgency=critical' if failed else '--urgency=normal',title,text],env=env,capture_output=True,text=True,timeout=10)
    print('notification',result.returncode,result.stderr,flush=True)


def power_check():
    """디스크·전원을 점검한다. 여유 20 GiB 미만 또는 비충전 배터리 25% 미만이면 중단한다."""
    if shutil.disk_usage(BASE).free<20*1024**3:raise RuntimeError('내장 SSD 여유 공간 20 GiB 미만')
    supplies=list(Path('/sys/class/power_supply').glob('*'))
    ac=any((p/'online').exists() and (p/'online').read_text().strip()=='1' for p in supplies)
    capacities=[int((p/'capacity').read_text()) for p in supplies if (p/'capacity').exists()]
    if not ac and capacities and min(capacities)<25:raise RuntimeError('충전기 미연결 및 배터리 25% 미만')
    return dict(ac_connected=ac,battery_percent=capacities,free_bytes=shutil.disk_usage(BASE).free)


def publish(summary):
    """기계 판독 결과와 브라우저에서 읽을 수 있는 로컬 요약을 저장한다."""
    save(OUT/'summary.json',summary)
    content='<html lang="ko"><meta charset="utf-8"><title>야간 자세 평가 실험</title><style>body{font-family:system-ui;max-width:1050px;margin:32px auto;background:#16232c;color:#eee}pre{white-space:pre-wrap}a{color:#73dccb}</style><h1>야간 자세 평가 실험</h1><p>AI 라벨은 미검토 초안입니다. 점수 모델 시험은 같은 AI 교사 초안과의 일치도이며 전문가 채점 정확도가 아닙니다.</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><p><a href="error_training_report.json">오류 모델 결과</a> · <a href="score_training_report.json">점수 모델 결과</a> · <a href="AI_labels.json">AI 라벨 초안</a></p></html>'
    (OUT/'summary.html').write_text(content,encoding='utf-8')


def start_stall_guard():
    """진행 파일이 5분간 갱신되지 않으면 실패 종료해 systemd 재개를 유도한다.

    Returns:
        백그라운드 감시 스레드. 대화 모델이나 외부 API를 호출하지 않는다.
    """
    def watch():
        while True:
            time.sleep(30)
            path=OUT/'status.json'
            if path.exists() and time.time()-path.stat().st_mtime>300:
                save(OUT/'stall_recovery.json',dict(reason='5분간 진행 갱신 없음',detected_at=time.time(),action='서비스 자동 재시작 및 캐시 재사용'))
                print('STALL_GUARD: restarting stalled worker',flush=True)
                os._exit(75)
    thread=threading.Thread(target=watch,name='progress-stall-guard',daemon=True)
    thread.start()
    return thread


def main():
    """잠금·전원 확인 후 네 단계를 실행하고 제한 사항을 포함한 결과를 저장한다."""
    # 품질 검토 중인 초안이 자동 재시작으로 학습에 들어가는 것을 차단한다.
    hold=OUT/'label_quality_hold.json'
    if hold.exists() and json.loads(hold.read_text()).get('hold'):
        print('LABEL_QUALITY_HOLD: 기존 초안 검증 전 자동 학습을 보류합니다.',flush=True)
        return
    lock=(OUT/'run.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('같은 야간 작업이 이미 실행 중')
    (OUT/'worker.pid').write_text(str(os.getpid()))
    start=time.time();summary=dict(status='running',pid=os.getpid(),AI_labels_reviewed=False,expert_validated=False,stages={},limitations=[])
    progress('preflight',**power_check())
    start_stall_guard()
    if '__EGL_VENDOR_LIBRARY_FILENAMES' not in os.environ:raise RuntimeError('RTX 실행 파일로 시작해야 합니다.')
    import prepare,teacher,features,train
    t=time.time();items,candidates=prepare.prepare();summary['stages']['candidate_metadata_preparation_seconds']=time.time()-t
    progress('score_candidates_prepared',clips=len(items),score_candidate_pool=len(candidates),target_score_clips_per_exercise=70)
    labels=[]
    try:
        t=time.time();teacher.start_server();labels=teacher.label_all(candidates);summary['stages']['AI_score_labeling_seconds']=time.time()-t
    except Exception as exc:
        traceback.print_exc();summary['limitations'].append('AI 초안 라벨링 오류: '+repr(exc))
        labels=[json.loads(p.read_text()) for p in (OUT/'score_labels').glob('*.json')]
        save(OUT/'AI_labels.json',labels)
    finally:
        try:
            teacher.request('/api/generate',{'model':teacher.MODEL,'keep_alive':0},timeout=60)
            teacher.request('/api/generate',{'model':'qwen3-vl:2b-instruct','keep_alive':0},timeout=60)
            for _ in range(15):
                if not teacher.request('/api/ps',timeout=5).get('models'):break
                time.sleep(2)
        except Exception as exc:print('teacher unload:',repr(exc),flush=True)
    progress('score_labeling_complete',eligible=sum(bool(r.get('eligible_for_experimental_training')) for r in labels))
    # 이영준 담당 점수 학습·평가를 먼저 마치고 독립 결과를 저장한다.
    power_check();t=time.time();features.extract_scores(candidates,labels);summary['stages']['score_feature_seconds']=time.time()-t
    t=time.time();score_report=train.train_scores(candidates,labels);summary['stages']['score_training_seconds']=time.time()-t
    summary['score_model_status']={ex:r['status'] for ex,r in score_report.items()}
    summary['status']='score_phase_complete_error_phase_pending'
    publish(summary)
    save(OUT/'score_phase_result.json',dict(status='experimental_score_phase_finished',expert_validated=False,models=summary['score_model_status']))
    progress('score_models_complete',models=summary['score_model_status'])
    # 점수 초안과 함께 관측한 오류는 재추론하지 않고 별도 라벨로 정리한다.
    save(OUT/'AI_error_labels.json',[dict(id=r['id'],exercise=r.get('exercise'),observed_errors=r.get('observed_errors',[]),label_source=r.get('label_source','local_VLM_visual_draft'),review_status='unreviewed',eligible=r.get('eligible_for_experimental_training',False)) for r in labels])
    progress('error_labeling_complete',provider_clips=len(items),AI_draft_clips=len(labels))
    power_check();t=time.time();features.extract_errors(items);summary['stages']['error_feature_seconds']=time.time()-t
    t=time.time();error_report=train.train_errors(items,labels);summary['stages']['error_training_seconds']=time.time()-t
    progress('error_models_complete',exercises=list(error_report))
    counts=collections.Counter(r.get('exercise') for r in labels if r.get('eligible_for_experimental_training'))
    summary['eligible_AI_labels_by_exercise']=dict(counts)
    for ex in ['pushups','squats','plank','ytw','shoulder']:
        if counts[ex]<70:summary['limitations'].append(f'{ex}: 유효 점수 초안 {counts[ex]}/70개. 나머지는 판단 불가·관측 부족 등으로 제외.')
    summary['score_model_status']={ex:r['status'] for ex,r in score_report.items()}
    summary['provider_error_models']={ex:dict(collections.Counter(v['status'] for v in r.get('provider_error_tasks',{}).values())) for ex,r in error_report.items()}
    summary['limitations'].append('점수와 AI 세부 오류는 로컬 시각 모델 초안이며 팀·전문가 검토 전이다. 실제 채점 정확도와 별도 검증이 필요하다.')
    summary.update(status='complete_with_limitations',elapsed_hours=(time.time()-start)/3600,source_data_preserved=True,existing_web_replaced=False)
    publish(summary);progress('complete_with_limitations',summary=str(OUT/'summary.html'))
    notify('야간 자세 평가 실험 완료','학습·제외 결과를 확인하세요. AI 초안은 미검토입니다.\n'+str(OUT/'summary.html'))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        traceback.print_exc();progress('failed',error=repr(exc),resume='동일 실행 명령으로 재개 가능. 기존 라벨·특징을 재사용함.')
        save(OUT/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        try:notify('야간 실험 확인 필요',str(exc)+'\n'+str(OUT/'pipeline.log'),True)
        except Exception:traceback.print_exc()
        raise SystemExit(1)
