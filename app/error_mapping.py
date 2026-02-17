import httpx
from fastapi import HTTPException
from tenacity import RetryError


def map_exception_to_http(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code if exc.response is not None else 502
        return HTTPException(status_code=status, detail=str(exc))
    if isinstance(exc, RetryError):
        status_error = _find_http_status_error(exc)
        if status_error is not None:
            status = status_error.response.status_code if status_error.response is not None else 502
            return HTTPException(status_code=status, detail=str(status_error))
        return HTTPException(status_code=503, detail="Upstream request retries exhausted")
    return HTTPException(status_code=500, detail=str(exc))


def _find_http_status_error(exc: BaseException) -> httpx.HTTPStatusError | None:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc

    if isinstance(exc, RetryError):
        last = exc.last_attempt.exception()
        if isinstance(last, BaseException):
            nested = _find_http_status_error(last)
            if nested is not None:
                return nested

    for candidate in (getattr(exc, "__cause__", None), getattr(exc, "__context__", None)):
        if isinstance(candidate, BaseException):
            nested = _find_http_status_error(candidate)
            if nested is not None:
                return nested

    return None
