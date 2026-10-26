"""인증 사용자의 완료 세션을 날짜별로 조회해 운동 리포트 응답을 만듭니다. 다른 사용자의 결과가 섞이지 않도록 조회 조건에 사용자 ID를 포함합니다."""

import json
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query

from Server.AIFitService.database.connection import get_connection
from Server.AIFitService.services.token_service import get_current_user

router = APIRouter(tags=["Exercise Reports"])


@router.get("/exercise-report")
def get_exercise_report(
    report_date: date = Query(
        ..., alias="date", description="운동 날짜 (YYYY-MM-DD)"
    ),
    current_user: dict = Depends(get_current_user),
):
    """Return the latest completed workout report for the authenticated user and date."""
    with get_connection() as conn:
        row = conn.execute(
            """SELECT report.report_id, report.session_id, report.avg_score,
                      report.report_json, report.created_at
               FROM ExerciseReport AS report
               JOIN ExerciseSession AS session
                 ON session.session_id = report.session_id
               WHERE session.user_id = ?
                 AND substr(session.started_at, 1, 10) = ?
                 AND session.session_status = 'COMPLETED'
               ORDER BY report.created_at DESC, report.report_id DESC
               LIMIT 1""",
            (current_user["user_id"], report_date.isoformat()),
        ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="해당 날짜의 완료된 운동 리포트가 없습니다.",
        )

    try:
        report_json = json.loads(row["report_json"])
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(
            status_code=500,
            detail="저장된 운동 리포트 JSON을 읽을 수 없습니다.",
        )

    return {
        "report_id": row["report_id"],
        "session_id": row["session_id"],
        "date": report_date.isoformat(),
        "avg_score": row["avg_score"],
        "created_at": row["created_at"],
        "report_json": report_json,
    }
