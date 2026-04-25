import asyncio
import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse

from app.schemas.api import ErrorResponse, MatchRequest, MatchResponse
from app.services.pipeline_errors import PipelineStageError
from app.services.pipeline_service import MatchPipelineService

logger = logging.getLogger(__name__)
router = APIRouter()


def get_match_pipeline_service() -> MatchPipelineService:
    return MatchPipelineService()


def format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=True)}\n\n"


def build_error_payload(
    *,
    stage: str,
    code: str,
    message: str,
) -> dict:
    payload = ErrorResponse(
        error={
            "code": code,
            "stage": stage,
            "message": message,
        }
    ).model_dump()
    payload["detail"] = message
    return payload


@router.post(
    "/match",
    response_model=MatchResponse,
    responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)
@router.post(
    "/match/",
    response_model=MatchResponse,
    include_in_schema=False,
    responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
)
async def match_candidates(
    payload: MatchRequest,
    pipeline_service: MatchPipelineService = Depends(get_match_pipeline_service),
) -> MatchResponse:
    try:
        if hasattr(pipeline_service, "run_async"):
            result = await pipeline_service.run_async(
                payload.job_description,
                top_k_search=payload.top_k_search,
                top_k_final=payload.top_k_final,
                page=payload.page,
                page_size=payload.page_size,
                include_outreach=payload.include_outreach,
            )
        else:
            result = pipeline_service.run(
                payload.job_description,
                top_k_search=payload.top_k_search,
                top_k_final=payload.top_k_final,
                page=payload.page,
                page_size=payload.page_size,
                include_outreach=payload.include_outreach,
            )
    except PipelineStageError as exc:
        logger.exception("Match pipeline failed at stage=%s", exc.stage)
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_payload(
                stage=exc.stage,
                code=exc.code,
                message=exc.message,
            ),
        )
    except Exception as exc:
        logger.exception("Match pipeline failed unexpectedly")
        return JSONResponse(
            status_code=500,
            content=build_error_payload(
                stage="match",
                code="match_failed",
                message=str(exc) or "The match pipeline failed unexpectedly.",
            ),
        )

    return MatchResponse(
        parsed_job_description=result.parsed_job_description,
        rankings=result.rankings,
        total_candidates_considered=result.total_candidates_ranked,
        total_candidates_retrieved=result.total_candidates_retrieved,
        total_candidates_returned=result.total_candidates_returned,
        page=result.page,
        page_size=result.page_size,
        total_pages=result.total_pages,
    )


@router.post("/match/stream")
@router.post("/match/stream/", include_in_schema=False)
async def stream_match_candidates(
    payload: MatchRequest,
    pipeline_service: MatchPipelineService = Depends(get_match_pipeline_service),
):
    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    async def progress_callback(event: dict) -> None:
        await queue.put(event)

    async def runner() -> None:
        try:
            result = await pipeline_service.run_async(
                payload.job_description,
                top_k_search=payload.top_k_search,
                top_k_final=payload.top_k_final,
                page=payload.page,
                page_size=payload.page_size,
                include_outreach=payload.include_outreach,
                progress_callback=progress_callback,
            )
            await queue.put(
                {
                    "event": "result",
                    "payload": MatchResponse(
                        parsed_job_description=result.parsed_job_description,
                        rankings=result.rankings,
                        total_candidates_considered=result.total_candidates_ranked,
                        total_candidates_retrieved=result.total_candidates_retrieved,
                        total_candidates_returned=result.total_candidates_returned,
                        page=result.page,
                        page_size=result.page_size,
                        total_pages=result.total_pages,
                    ).model_dump(),
                }
            )
        except Exception as exc:
            logger.exception("Streaming match pipeline failed")
            if isinstance(exc, PipelineStageError):
                error_payload = build_error_payload(
                    stage=exc.stage,
                    code=exc.code,
                    message=exc.message,
                )
            else:
                error_payload = build_error_payload(
                    stage="match",
                    code="match_failed",
                    message=str(exc) or "The streaming match pipeline failed unexpectedly.",
                )
            await queue.put({"event": "error", "payload": error_payload})
        finally:
            await queue.put(None)

    async def event_stream():
        task = asyncio.create_task(runner())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break

                event_name = item.get("event", "progress")
                payload = item.get("payload", item)
                yield format_sse(event_name, payload)
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
