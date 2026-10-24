"""명시적 예약과 화면 연결이 있을 때만 준비 후 한 번 자동 시작한다."""
import math
import uuid


class StartReservation:
    def __init__(self):
        self.request = None
        self.state = 'idle'
        self.reason = None
        self.remaining = None

    def arm(self, exercise, view, now, probe):
        self.request = dict(id=uuid.uuid4().hex,exercise=exercise,view=view,
                            armed_at=now,heartbeat_at=now,countdown_at=None)
        self.probe = probe
        self.state,self.reason,self.remaining = 'waiting',None,None
        return self.status()

    def cancel(self, reason):
        self.request = None
        self.state,self.reason,self.remaining = 'cancelled',reason,None

    def heartbeat(self, request_id, now):
        if self.request and self.request['id'] == request_id:
            self.request['heartbeat_at'] = now

    def step(self, now, ready):
        r = self.request
        if not r:
            return None
        if now-r['heartbeat_at'] > 4:
            self.cancel('화면 연결이 끊겨 시작 예약을 취소했습니다.')
            return None
        if now-r['armed_at'] > 180:
            self.cancel('3분 동안 준비되지 않아 시작 예약을 취소했습니다.')
            return None
        if not ready:
            r['countdown_at'] = None
            self.state,self.remaining = 'waiting',None
            return None
        if r['countdown_at'] is None:
            r['countdown_at'] = now
        self.remaining = max(0,math.ceil(3-(now-r['countdown_at'])))
        self.state = 'countdown'
        if now-r['countdown_at'] < 3:
            return None
        event = {k:r[k] for k in ('id','exercise','view')}
        self.request = None
        self.state,self.remaining = 'started',None
        return event

    def status(self):
        r = self.request or {}
        return dict(state=self.state,request_id=r.get('id'),exercise=r.get('exercise'),
                    view=r.get('view'),remaining=self.remaining,reason=self.reason)
