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
    def test_key_error_maps_to_400(self):
        result = map_exception_to_http(KeyError("missing_key"))
        assert result.status_code == 400

    def test_value_error_maps_to_400(self):
        result = map_exception_to_http(ValueError("invalid value"))
        assert result.status_code == 400

    def test_http_status_error_uses_response_status(self):
        result = map_exception_to_http(_http_error(404))
        assert result.status_code == 404

    def test_http_status_error_with_none_response_maps_to_502(self):
        exc = httpx.HTTPStatusError("error", request=MagicMock(), response=None)
        result = map_exception_to_http(exc)
        assert result.status_code == 502

    def test_retry_error_without_http_error_maps_to_503(self):
        result = map_exception_to_http(_retry_error(RuntimeError("connection refused")))
        assert result.status_code == 503
        assert "exhausted" in result.detail

    def test_retry_error_wrapping_http_error_passes_through_status(self):
        result = map_exception_to_http(_retry_error(_http_error(502)))
        assert result.status_code == 502

    def test_generic_exception_maps_to_500(self):
        result = map_exception_to_http(RuntimeError("unexpected"))
        assert result.status_code == 500


class TestFindHttpStatusError:
    def test_finds_direct_http_status_error(self):
        exc = _http_error(404)
        assert _find_http_status_error(exc) is exc

    def test_returns_none_for_plain_exception(self):
        assert _find_http_status_error(ValueError("nope")) is None

    def test_finds_error_via_cause_chain(self):
        http_exc = _http_error(503)
        wrapper = RuntimeError("wrapper")
        wrapper.__cause__ = http_exc
        assert _find_http_status_error(wrapper) is http_exc

    def test_finds_error_inside_retry_error(self):
        http_exc = _http_error(429)
        result = _find_http_status_error(_retry_error(http_exc))
        assert result is http_exc
