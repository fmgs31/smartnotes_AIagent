import pytest
import httpx
from unittest.mock import MagicMock
from tenacity import RetryError

from app.error_mapping import map_exception_to_http, _find_http_status_error


def _http_error(status: int) -> httpx.HTTPStatusError:
    mock_response = MagicMock()
    mock_response.status_code = status
    return httpx.HTTPStatusError("error", request=MagicMock(), response=mock_response)


def _retry_error(inner: Exception) -> RetryError:
    attempt = MagicMock()
    attempt.exception.return_value = inner
    err = RetryError(attempt)
    err.__cause__ = inner
    return err


class TestMapExceptionToHttp:
    """map_exception_to_http: translates internal exceptions into user-facing HTTPExceptions."""

    def test_key_error_maps_to_400(self):
        # KeyError usually means a missing request field → 400 bad request.
        result = map_exception_to_http(KeyError("missing_key"))
        assert result.status_code == 400, (
            f"Expected 400 for KeyError, got {result.status_code}: {result.detail!r}"
        )

    def test_value_error_maps_to_400(self):
        # ValueError means bad input shape/value → 400 bad request.
        result = map_exception_to_http(ValueError("invalid value"))
        assert result.status_code == 400, (
            f"Expected 400 for ValueError, got {result.status_code}: {result.detail!r}"
        )

    def test_http_status_error_uses_response_status(self):
        # An httpx error carrying a response must forward that upstream status code.
        result = map_exception_to_http(_http_error(404))
        assert result.status_code == 404, (
            f"Expected upstream status 404 to pass through, got {result.status_code}: {result.detail!r}"
        )

    def test_http_status_error_with_none_response_maps_to_502(self):
        # HTTPStatusError without a response → upstream gateway issue → 502.
        exc = httpx.HTTPStatusError("error", request=MagicMock(), response=None)
        result = map_exception_to_http(exc)
        assert result.status_code == 502, (
            f"Expected 502 for HTTPStatusError with response=None, got {result.status_code}: {result.detail!r}"
        )

    def test_retry_error_without_http_error_maps_to_503(self):
        # Retries exhausted on a non-HTTP error → the service was unavailable → 503.
        result = map_exception_to_http(_retry_error(RuntimeError("connection refused")))
        assert result.status_code == 503, (
            f"Expected 503 when retries are exhausted on a transient error, "
            f"got {result.status_code}: {result.detail!r}"
        )
        assert "exhausted" in result.detail, (
            f"Expected 'exhausted' in detail for a RetryError, got {result.detail!r}"
        )

    def test_retry_error_wrapping_http_error_passes_through_status(self):
        # If the RetryError wraps an HTTPStatusError, the original upstream status wins over 503.
        result = map_exception_to_http(_retry_error(_http_error(502)))
        assert result.status_code == 502, (
            f"Expected wrapped HTTP status 502 to pass through the RetryError, "
            f"got {result.status_code}: {result.detail!r}"
        )

    def test_generic_exception_maps_to_500(self):
        # Catch-all for unexpected exceptions.
        result = map_exception_to_http(RuntimeError("unexpected"))
        assert result.status_code == 500, (
            f"Expected 500 for an unexpected RuntimeError, got {result.status_code}: {result.detail!r}"
        )


class TestFindHttpStatusError:
    """_find_http_status_error: walks __cause__/RetryError chains looking for an HTTPStatusError."""

    def test_finds_direct_http_status_error(self):
        exc = _http_error(404)
        result = _find_http_status_error(exc)
        assert result is exc, f"Expected direct HTTPStatusError to be returned, got {result!r}"

    def test_returns_none_for_plain_exception(self):
        result = _find_http_status_error(ValueError("nope"))
        assert result is None, f"Expected None for non-HTTP exception, got {result!r}"

    def test_finds_error_via_cause_chain(self):
        # Wrapper exception with __cause__ set to an HTTPStatusError → must unwrap one level.
        http_exc = _http_error(503)
        wrapper = RuntimeError("wrapper")
        wrapper.__cause__ = http_exc
        result = _find_http_status_error(wrapper)
        assert result is http_exc, (
            f"Expected the inner HTTPStatusError via __cause__, got {result!r}"
        )

    def test_finds_error_inside_retry_error(self):
        # RetryError from tenacity also hides the underlying exception — must be unwrapped.
        http_exc = _http_error(429)
        result = _find_http_status_error(_retry_error(http_exc))
        assert result is http_exc, (
            f"Expected the inner HTTPStatusError via RetryError, got {result!r}"
        )
