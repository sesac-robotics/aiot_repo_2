from fastapi import FastAPI, Request
from userLogin import router as login_router

app = FastAPI(title="AIFit Server")

# 회원가입 및 로그인 라우터 등록
app.include_router(login_router)


# 서버 상태 확인 API
@app.get("/health")
def health_check():
    return {"status": "ok"}


# GUI 및 다른 클라이언트의 요청 로그 출력
@app.middleware("http")
async def log_gui_connection(request: Request, call_next):
    client_ip = request.client.host if request.client else "알 수 없음"

    print(
        f"[요청 수신] 클라이언트 IP: {client_ip} | "
        f"요청: {request.method} {request.url.path}",
        flush=True
    )

    response = await call_next(request)

    print(
        f"[응답 완료] {request.method} {request.url.path} "
        f"| 상태 코드: {response.status_code}",
        flush=True
    )

    return response
