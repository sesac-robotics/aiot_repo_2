"""공유 SessionManager를 호출하는 내부 제어 API. DB/분석기 호출은 하지 않습니다."""
import os
import secrets
from typing import Annotated
from dotenv import load_dotenv

from fastapi import APIRouter, Depends, HTTPException, Path, Request, Security
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from fastapi.security import APIKeyHeader

from app.schemas.session import ExerciseStartRequest, SessionLiveResponse
from app.session import (
    ExerciseNotFound, InvalidSessionInput, SessionConflict, SessionManager, SessionNotFound,
)

API_KEY_HEADER = "X-Internal-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def verify_api_key(request: Request):
    # 미설정 시 fail-closed. 키는 로그/응답에 포함하지 않습니다.
    expected = getattr(request.app.state, 'internal_api_key', None)
    if not expected or not expected.strip():
        raise HTTPException(503, '내부 API 키가 설정되지 않았습니다.')
    supplied = request.headers.get(API_KEY_HEADER)
    if supplied is None or not secrets.compare_digest(supplied.encode('utf-8'), expected.encode('utf-8')):
        raise HTTPException(401, '내부 API 인증에 실패했습니다.')


class InternalSessionRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def handle(request: Request):
            # Body 파싱보다 먼저 인증하며 400 변환은 이 라우터에만 한정합니다.
            verify_api_key(request)
            try:
                return await handler(request)
            except RequestValidationError as error:
                raise HTTPException(400, '요청 경로 또는 JSON Body가 올바르지 않습니다.') from error
            except InvalidSessionInput as error:
                raise HTTPException(400, str(error)) from error
            except (SessionNotFound, ExerciseNotFound) as error:
                raise HTTPException(404, str(error)) from error
            except SessionConflict as error:
                raise HTTPException(409, str(error)) from error

        return handle


router = APIRouter(
    prefix='/internal/sessions', tags=['internal sessions'], route_class=InternalSessionRoute,
    # OpenAPI에도 같은 API 키 Header 규격을 노출합니다. 실제 인증은 위에서 먼저 수행합니다.
    dependencies=[Security(api_key_header)],
    responses={400: {'description': '잘못된 요청'}, 401: {'description': '인증 실패'},
               404: {'description': '현재 세션 또는 운동 없음'}, 409: {'description': '상태 충돌'},
               503: {'description': '내부 API 키 미설정'}},
)


def get_session_manager(request: Request) -> SessionManager:
    # 요청마다 새 관리자를 만들지 않습니다. 향후 추론 워커에도 이 인스턴스를 전달합니다.
    return request.app.state.session_manager


Manager = Annotated[SessionManager, Depends(get_session_manager)]
PositiveID = Annotated[int, Path(gt=0)]


@router.post('/{session_id}/start', response_model=SessionLiveResponse)
def start_session(session_id: PositiveID, manager: Manager):
    return manager.start_session(session_id)


@router.post('/{session_id}/exercises/{exercise_id}/start', response_model=SessionLiveResponse)
def start_exercise(session_id: PositiveID, exercise_id: PositiveID,
                   body: ExerciseStartRequest, manager: Manager):
    with manager.target_session(session_id):
        return manager.start_exercise(exercise_id, body.exercise_name, body.order_no, body.target_reps)


@router.post('/{session_id}/pause', response_model=SessionLiveResponse)
def pause_session(session_id: PositiveID, manager: Manager):
    with manager.target_session(session_id):
        return manager.pause_session()


@router.post('/{session_id}/resume', response_model=SessionLiveResponse)
def resume_session(session_id: PositiveID, manager: Manager):
    with manager.target_session(session_id):
        return manager.resume_session()


@router.post('/{session_id}/exercises/{exercise_id}/finish', response_model=SessionLiveResponse)
def finish_exercise(session_id: PositiveID, exercise_id: PositiveID, manager: Manager):
    with manager.target_session(session_id, exercise_id):
        return manager.finish_exercise()


@router.post('/{session_id}/finish', response_model=SessionLiveResponse)
def finish_session(session_id: PositiveID, manager: Manager):
    with manager.target_session(session_id):
        return manager.finish_session()


@router.get('/{session_id}/live', response_model=SessionLiveResponse)
def live_session(session_id: PositiveID, manager: Manager):
    with manager.target_session(session_id):
        return manager.get_snapshot()
