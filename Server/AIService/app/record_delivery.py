"""반복별 records 전송. 영상/세션 잠금 안에서는 네트워크를 호출하지 않습니다."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import queue
import threading
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError
from urllib.parse import urlsplit

LOGGER = logging.getLogger('uvicorn.error')


def record_payload(exercise, posture):
    classification = (posture or {}).get('classification') or {}
    state = classification.get('status')
    pose_state = {'NORMAL':'CORRECT', 'ERROR':'INCORRECT'}.get(state, 'UNKNOWN')
    errors = ((posture or {}).get('posture_errors') or []) if pose_state == 'INCORRECT' else []
    codes = [item['code'] for item in errors if item.get('code')]
    mapping = {'SQUAT_KNEE_VALGUS':'KNEE_VALGUS'}
    payload = dict(exercise_id=exercise['exercise_id'], set_no=exercise['set_no'],
                   rep_no=exercise['rep_count'], pose_state=pose_state, score=exercise['score'],
                   error_codes=[mapping.get(code,code) for code in codes])
    if pose_state == 'INCORRECT' and errors:
        payload['feedback_code'] = 'KNEE_ALIGNMENT' if 'SQUAT_KNEE_VALGUS' in codes else codes[0]
        text = errors[0].get('feedback')
        if text:payload['feedback_text'] = text
    return payload


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # 인증 헤더를 다른 주소로 전달하지 않습니다.
        return None


class RecordDelivery:
    def __init__(self, url_template, api_key, *, timeout=3., failure_path=None):
        parsed = urlsplit(url_template)
        if parsed.scheme not in ('http','https') or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError('AIFIT_RECORDS_URL must be an HTTP(S) endpoint')
        if not api_key or not api_key.strip():
            raise ValueError('records delivery requires X-Internal-API-Key')
        # format의 다른 플레이스홀더/잘못된 괄호는 시작 시 거부합니다.
        url_template.format(session_id=1)
        self.url_template,self.api_key,self.timeout = url_template,api_key,timeout
        self.failure_path = Path(failure_path) if failure_path else Path(__file__).resolve().parents[1]/'logs'/'records_failed.jsonl'
        self.queue = queue.Queue(maxsize=1000)
        self.stop_event = threading.Event()
        self.thread = None
        self.opener = build_opener(NoRedirect())
        self.failure_lock = threading.Lock()

    @classmethod
    def from_env(cls):
        url = os.getenv('AIFIT_RECORDS_URL','').strip()
        if not url:
            LOGGER.info('[Records] disabled: AIFIT_RECORDS_URL not set')
            return None
        return cls(url,os.getenv('AIFIT_INTERNAL_API_KEY') or os.getenv('INTERNAL_API_KEY'))

    def start(self):
        self.thread = threading.Thread(target=self._run,name='records-delivery',daemon=True)
        self.thread.start()
        LOGGER.info('[Records] delivery worker started')

    def submit(self, session_id, payload):
        # 호출 시점의 값을 고정하여 다음 반복/세션 변경과 분리합니다.
        item = dict(session_id=session_id,payload=deepcopy(payload))
        try:self.queue.put_nowait(item)
        except queue.Full:
            self._failed(item,'queue_full')

    def close(self):
        self.stop_event.set()
        if self.thread:self.thread.join(timeout=self.timeout+1.)
        # 종료 시 남은 미전송 요청은 저장하고 진행 중인 요청은 워커가 마무리합니다.
        while True:
            try:item=self.queue.get_nowait()
            except queue.Empty:break
            self._failed(item,'shutdown_pending')
            self.queue.task_done()

    def _failed(self,item,reason):
        LOGGER.error('[Records] delivery failed session=%s rep=%s reason=%s',
                     item['session_id'],item['payload']['rep_no'],reason)
        try:
            self.failure_path.parent.mkdir(parents=True,exist_ok=True)
            with self.failure_lock, self.failure_path.open('a',encoding='utf-8') as file:
                file.write(json.dumps(dict(logged_at=datetime.now(timezone.utc).isoformat(),
                                           reason=reason,**item),ensure_ascii=False,allow_nan=False)+'\n')
        except OSError:
            LOGGER.exception('[Records] could not retain failed payload')

    def _run(self):
        while not self.stop_event.is_set():
            try:item=self.queue.get(timeout=.2)
            except queue.Empty:continue
            try:
                url=self.url_template.format(session_id=item['session_id'])
                request=Request(url,data=json.dumps(item['payload'],ensure_ascii=False,allow_nan=False).encode('utf-8'),
                                headers={'Content-Type':'application/json','X-Internal-API-Key':self.api_key},method='POST')
                with self.opener.open(request,timeout=self.timeout) as response:
                    if not 200 <= response.status < 300:raise HTTPError(url,response.status,'non-success',None,None)
                LOGGER.info('[Records] saved session=%s exercise=%s rep=%s',item['session_id'],
                            item['payload']['exercise_id'],item['payload']['rep_no'])
            except HTTPError as error:
                self._failed(item,'http_'+str(error.code))
            except Exception as error:
                # 타임아웃 이후 재전송 시 DB 중복 가능성이 있어 실패 요청을 보관합니다.
                self._failed(item,type(error).__name__)
            finally:self.queue.task_done()
