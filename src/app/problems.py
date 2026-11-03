from typing import Callable, Optional

from fastapi import Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel

PROBLEM_TYPE_BASE = "https://mrx.internal/problems"
VALUE_ERROR_PREFIX = "Value error, "


class Problem(BaseModel):
    """RFC 9457 Problem Details"""
    type: str
    title: str
    status: int
    detail: Optional[str] = None
    instance: Optional[str] = None


class ProblemError(Exception):
    def __init__(self, status: int, code: str, title: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.title = title
        self.detail = detail


def not_found(detail: str) -> ProblemError:
    return ProblemError(404, "not-found", "리소스가 없음", detail)


def conflict(detail: str) -> ProblemError:
    return ProblemError(409, "conflict", "요청 충돌", detail)


def not_implemented(detail: str) -> ProblemError:
    return ProblemError(501, "not-implemented", "아직 지원하지 않음", detail)


def _response(request: Request, status: int, code: str, title: str, detail: str) -> JSONResponse:
    problem = Problem(
        type=f"{PROBLEM_TYPE_BASE}/{code}",
        title=title,
        status=status,
        detail=detail,
        instance=request.url.path,
    )
    return JSONResponse(status_code=status, content=problem.model_dump())


def _validation_detail(exc: RequestValidationError) -> str:
    messages = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"][1:])
        message = error["msg"].removeprefix(VALUE_ERROR_PREFIX)
        messages.append(f"{location}: {message}" if location else message)
    return "; ".join(messages)


class ProblemRoute(APIRoute):
    """Answers errors as Problem Details, and request errors as 400 instead of FastAPI's 422.

    Set per router, so the routers that are not on the v1 spec yet keep their current errors.
    """

    def get_route_handler(self) -> Callable:
        handler = super().get_route_handler()

        async def problem_route_handler(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as exc:
                return _response(request, 400, "invalid-request", "요청 형식 오류", _validation_detail(exc))
            except ProblemError as exc:
                return _response(request, exc.status, exc.code, exc.title, exc.detail)

        return problem_route_handler
