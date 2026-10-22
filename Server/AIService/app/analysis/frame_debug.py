"""YTW_DEBUG에서만 영상·원본·분석 입력의 출처를 연결합니다. 사람 선택/판정은 바꾸지 않습니다."""
import hashlib
import json
import logging
import math
import os
import time

_FIELDS=('x','y','z','visibility','presence')


def _safe(value):
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    if isinstance(value,dict):return {k:_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_safe(v) for v in value]
    return value


def fingerprint(points):
    # 순서와 ID까지 포함하므로 좌표/신뢰도 외에 인덱스 이동도 검출합니다.
    data=[{k:p.get(k) for k in ('id',*_FIELDS)} for p in points]
    return hashlib.sha256(json.dumps(_safe(data),sort_keys=True,default=str).encode()).hexdigest()


def _center(points,size):
    by_id={p['id']:p for p in points}
    if any(i not in by_id for i in (11,12)):return None
    x=(by_id[11]['x']+by_id[12]['x'])/2
    y=(by_id[11]['y']+by_id[12]['y'])/2
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (x,y)):return None
    return dict(normalized=[x,y],pixel=[x*size[0],y*size[1]])


def compare(source,target):
    """원본 정규화 값을 직접 비교하며 각도·자세를 다시 계산하지 않습니다."""
    expected={p['id']:p for p in source}
    actual={p.get('id'):p for p in target}
    mismatches=[]
    for ident in sorted(set(expected)|set(actual)):
        if ident not in expected or ident not in actual:
            mismatches.append(dict(id=ident,field='id',reason='missing_or_extra_joint'))
            continue
        for field in _FIELDS:
            a,b=expected[ident].get(field),actual[ident].get(field)
            # NaN끼리도 정상적인 일치로 표시하지 않습니다.
            if a!=b:mismatches.append(dict(id=ident,field=field,source=a,target=b))
    duplicate_ids=len(actual)!=len(target)
    return dict(values_equal=not mismatches and not duplicate_ids,
                order_equal=[p.get('id') for p in source]==[p.get('id') for p in target],
                duplicate_ids=duplicate_ids,mismatches=mismatches)


class FrameDebug:
    def __init__(self):
        self.enabled=os.getenv('YTW_DEBUG')=='1'
        self._last={}
        self._previous_center=None

    def emit(self,kind,payload):
        if not self.enabled:return
        try:
            now=time.monotonic()
            if now-self._last.get(kind,-math.inf)<1.:return
            logging.getLogger('uvicorn.error').info('[YTWDebug] %s',json.dumps(
                _safe(dict(kind=kind,**payload)),ensure_ascii=False,default=str,allow_nan=False))
            self._last[kind]=now
        except Exception:
            pass

    def capture(self,result,keypoints,*,frame_id,timestamp,size,estimator,config,config_source):
        if not self.enabled:return None
        try:
            return self._capture(result,keypoints,frame_id=frame_id,timestamp=timestamp,size=size,
                                 estimator=estimator,config=config,config_source=config_source)
        except Exception as error:
            self.emit('frame_trace_error',dict(frame_id=frame_id,timestamp=timestamp,reason=str(error)))
            return None  # 진단 실패로 기존 프레임을 잃지 않도록 합니다.

    def _capture(self,result,keypoints,*,frame_id,timestamp,size,estimator,config,config_source):
        poses=result.pose_landmarks
        raw=[]
        if poses:
            raw=[dict(id=i,**{field:getattr(p,field,None) for field in _FIELDS}) for i,p in enumerate(poses[0])]
        center=_center(raw,size)
        delta=None
        if center is not None and self._previous_center is not None:
            delta=math.dist(center['normalized'],self._previous_center)
        self._previous_center=center['normalized'] if center else None
        return dict(frame_id=frame_id,timestamp=timestamp,frame_size=list(size),
                    mediapipe_timestamp_ms=getattr(estimator,'last_timestamp_ms',None),
                    configured_num_poses=getattr(estimator,'num_poses',None),
                    returned_num_poses=len(poses),selected_pose_index=0 if poses else None,
                    drawn_pose_indices=list(range(len(poses))),
                    shoulder_center=center,shoulder_center_delta=delta,
                    returned_shoulder_centers=[_center([dict(id=i,x=p.x,y=p.y) for i,p in enumerate(pose)],size)
                                               for pose in poses],
                    identity_verified=False,selection_policy='first_result_no_identity_lock',
                    renderer_min_visibility=getattr(estimator,'min_visibility',None),
                    analyzer_min_visibility=config.min_visibility,analyzer_min_presence=config.min_presence,
                    validation_config_source=config_source,
                    raw_fingerprint=fingerprint(raw),extracted_fingerprint=fingerprint(keypoints),
                    raw_to_extracted=compare(raw,keypoints),raw_points=raw,
                    extracted_points=[dict(p) for p in keypoints])

    @staticmethod
    def summary(trace):
        return {k:v for k,v in trace.items() if k not in ('raw_points','extracted_points')}

    def publish(self,trace):
        if trace is not None:
            self.emit('frame_result',self.summary(trace))

    def analyzer_input(self,points,trace,*,frame_id,timestamp):
        if not self.enabled or trace is None:return
        source=trace['raw_points'];extracted=trace['extracted_points']
        def by_id(items,ident):return next((p for p in items if p.get('id')==ident),None)
        self.emit('frame_input',dict(frame_id=frame_id,timestamp=timestamp,
            source_frame_id=trace['frame_id'],source_timestamp=trace['timestamp'],
            metadata_equal=frame_id==trace['frame_id'] and timestamp==trace['timestamp'],
            raw_to_analyzer=compare(source,points),extracted_to_analyzer=compare(extracted,points),
            analyzer_fingerprint=fingerprint(points),
            joints={str(i):dict(raw=by_id(source,i),extracted=by_id(extracted,i),analyzer=by_id(points,i))
                    for i in range(11,17)}))

    def overlay(self,frame,trace):
        if not self.enabled or trace is None:return
        try:
            import cv2
            width,height=trace['frame_size']
            scale=max(.3,min(.55,width/1000))
            line=max(12,int(22*scale/.55))
            def text(value,xy,color=(255,255,0)):
                cv2.putText(frame,value,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),3)
                cv2.putText(frame,value,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1)
            text(f"fid={trace['frame_id']} ts={trace['timestamp']:.6f} pose={trace['selected_pose_index']}",(6,50))
            center=trace['shoulder_center']
            center_text=('norm=('+','.join(f'{v:.4f}' for v in center['normalized'])+') px=('
                         +','.join(f'{v:.1f}' for v in center['pixel'])+')') if center else 'None'
            text('shoulder='+center_text,(6,50+line))
            counts={p['id']:sum(q['id']==p['id'] for q in trace['extracted_points']) for p in trace['extracted_points']}
            for n,ident in enumerate(range(11,17)):
                p=next((p for p in trace['extracted_points'] if p['id']==ident),None)
                def finite(v):return type(v) in (int,float) and math.isfinite(v)
                valid=(p is not None and counts[ident]==1
                       and all(finite(p.get(a)) and 0<=p[a]<=1 for a in ('x','y'))
                       and finite(p.get('visibility')) and trace['analyzer_min_visibility']<=p['visibility']<=1
                       and (p.get('presence') is None or finite(p['presence'])
                            and trace['analyzer_min_presence']<=p['presence']<=1))
                visibility=f"{p['visibility']:.3f}" if p and finite(p.get('visibility')) else 'None'
                label=f'{ident} v={visibility} YTWvalid={int(valid)}'
                color=(0,255,0) if valid else (0,0,255)
                text(label,(6,50+(2+n)*line),color)
                if p and all(finite(p.get(a)) and 0<=p[a]<=1 for a in ('x','y')):
                    xy=(min(width-1,int(p['x']*width)),min(height-1,int(p['y']*height)))
                    cv2.circle(frame,xy,5,color,1)
                    text(label,(min(xy[0],max(0,width-140)),max(line,min(height-line,xy[1]-6))),color)
        except Exception:
            pass  # 추가 오버레이 실패로 기존 스트리밍을 중단하지 않습니다.
