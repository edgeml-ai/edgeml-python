"""Tests for octomil.errors — canonical error codes and unified exception."""

from __future__ import annotations

import pytest

from octomil.errors import OctomilError, OctomilErrorCode

# ---------------------------------------------------------------------------
# Canonical codes — derived from the OctomilErrorCode enum so this list
# stays in lockstep with contract bumps. The hardcoded subset below
# (CORE_CODES) covers the v1.25.0 era for regression coverage.
# ---------------------------------------------------------------------------

ALL_CODES = [c.value for c in OctomilErrorCode]

CORE_CODES = [
    "invalid_api_key",
    "authentication_failed",
    "forbidden",
    "insufficient_scope",
    "missing_org_context",
    "device_not_registered",
    "token_expired",
    "device_revoked",
    "network_unavailable",
    "request_timeout",
    "server_error",
    "rate_limited",
    "invalid_input",
    "unsupported_modality",
    "context_too_large",
    "model_not_found",
    "no_default_model",
    "capability_not_supported",
    "previous_response_not_found",
    "app_not_found",
    "capability_not_configured",
    "app_context_conflict",
    "invalid_model_ref",
    "model_disabled",
    "version_not_found",
    "download_failed",
    "checksum_mismatch",
    "insufficient_storage",
    "insufficient_memory",
    "runtime_unavailable",
    "accelerator_unavailable",
    "model_load_failed",
    "inference_failed",
    "provider_error",
    "upstream_provider_error",
    "too_many_tools",
    "unsupported_tool_calling",
    "stream_interrupted",
    "policy_denied",
    "cloud_fallback_disallowed",
    "cloud_inference_not_allowed",
    "hosted_tts_disabled",
    "plan_limit_exceeded",
    "cloud_credentials_missing",
    "cloud_credentials_revoked",
    "cloud_provider_auth_failed",
    "max_tool_rounds_exceeded",
    "training_failed",
    "training_not_supported",
    "weight_upload_failed",
    "control_sync_failed",
    "assignment_not_found",
    "incident_not_found",
    "deployment_not_found",
    "experiment_not_found",
    "experiment_state_invalid",
    "api_key_not_found",
    "api_key_already_revoked",
    "integration_not_found",
    "billing_customer_not_found",
    "action_not_found",
    "action_state_invalid",
    "cancelled",
    "app_backgrounded",
    "unknown",
]


class TestOctomilErrorCodeEnum:
    def test_has_exactly_85_members(self) -> None:
        # Contract is at 1.27.0 — 85 canonical codes. The 1.27.0 batch
        # added passkeys + account-linking + admin codes. Update this
        # alongside contract bumps.
        assert len(OctomilErrorCode) == 85

    @pytest.mark.parametrize("value", ALL_CODES)
    def test_all_canonical_codes_exist(self, value: str) -> None:
        code = OctomilErrorCode(value)
        assert code.value == value

    def test_is_str_enum(self) -> None:
        """Each member should be usable as a plain string."""
        assert isinstance(OctomilErrorCode.UNKNOWN, str)
        assert OctomilErrorCode.UNKNOWN == "unknown"


# ---------------------------------------------------------------------------
# Retryable property
# ---------------------------------------------------------------------------

RETRYABLE_CODES = {
    OctomilErrorCode.NETWORK_UNAVAILABLE,
    OctomilErrorCode.REQUEST_TIMEOUT,
    OctomilErrorCode.SERVER_ERROR,
    OctomilErrorCode.RATE_LIMITED,
    OctomilErrorCode.DOWNLOAD_FAILED,
    OctomilErrorCode.CHECKSUM_MISMATCH,
    OctomilErrorCode.MODEL_LOAD_FAILED,
    OctomilErrorCode.INFERENCE_FAILED,
    OctomilErrorCode.STREAM_INTERRUPTED,
    OctomilErrorCode.CONTROL_SYNC_FAILED,
    OctomilErrorCode.APP_BACKGROUNDED,
    OctomilErrorCode.TRAINING_FAILED,
    OctomilErrorCode.WEIGHT_UPLOAD_FAILED,
    # New code added in v1.25.0 catalog bump; backoff_safe => retryable
    OctomilErrorCode.UPSTREAM_PROVIDER_ERROR,
}

NON_RETRYABLE_CODES = set(OctomilErrorCode) - RETRYABLE_CODES


class TestRetryableProperty:
    @pytest.mark.parametrize("code", sorted(RETRYABLE_CODES, key=lambda c: c.value))
    def test_retryable_codes(self, code: OctomilErrorCode) -> None:
        err = OctomilError(code=code, message="test")
        assert err.retryable is True

    @pytest.mark.parametrize("code", sorted(NON_RETRYABLE_CODES, key=lambda c: c.value))
    def test_non_retryable_codes(self, code: OctomilErrorCode) -> None:
        err = OctomilError(code=code, message="test")
        assert err.retryable is False


# ---------------------------------------------------------------------------
# from_http_status
# ---------------------------------------------------------------------------


class TestFromHttpStatus:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (400, OctomilErrorCode.INVALID_INPUT),
            (401, OctomilErrorCode.AUTHENTICATION_FAILED),
            (403, OctomilErrorCode.FORBIDDEN),
            (404, OctomilErrorCode.MODEL_NOT_FOUND),
            (429, OctomilErrorCode.RATE_LIMITED),
            (500, OctomilErrorCode.SERVER_ERROR),
            (502, OctomilErrorCode.SERVER_ERROR),
            (503, OctomilErrorCode.SERVER_ERROR),
            # Cutover follow-up #70: serve layer now emits 413/422/
            # 499/504/507 explicitly; the reverse map gains entries
            # so HTTP-consuming SDK paths get bounded codes back.
            (413, OctomilErrorCode.CONTEXT_TOO_LARGE),
            (422, OctomilErrorCode.UNSUPPORTED_MODALITY),
            (499, OctomilErrorCode.CANCELLED),
            (504, OctomilErrorCode.REQUEST_TIMEOUT),
            (507, OctomilErrorCode.INSUFFICIENT_STORAGE),
        ],
    )
    def test_mapped_status_codes(self, status: int, expected: OctomilErrorCode) -> None:
        err = OctomilError.from_http_status(status)
        assert err.code is expected

    @pytest.mark.parametrize("status", [200, 201, 204, 301, 408, 418])
    def test_unmapped_status_returns_unknown(self, status: int) -> None:
        err = OctomilError.from_http_status(status)
        assert err.code is OctomilErrorCode.UNKNOWN


# ---------------------------------------------------------------------------
# OctomilError construction and properties
# ---------------------------------------------------------------------------


class TestOctomilError:
    def test_is_exception_subclass(self) -> None:
        assert issubclass(OctomilError, Exception)

    def test_basic_construction(self) -> None:
        err = OctomilError(
            code=OctomilErrorCode.INVALID_API_KEY,
            message="bad key",
        )
        assert err.code is OctomilErrorCode.INVALID_API_KEY
        assert err.error_message == "bad key"
        assert str(err) == "bad key"
        assert err.cause is None

    def test_construction_with_cause(self) -> None:
        cause = ConnectionError("socket closed")
        err = OctomilError(
            code=OctomilErrorCode.NETWORK_UNAVAILABLE,
            message="connection lost",
            cause=cause,
        )
        assert err.cause is cause

    def test_retryable_delegates_to_code(self) -> None:
        retryable_err = OctomilError(
            code=OctomilErrorCode.SERVER_ERROR,
            message="500",
        )
        assert retryable_err.retryable is True

        non_retryable_err = OctomilError(
            code=OctomilErrorCode.FORBIDDEN,
            message="403",
        )
        assert non_retryable_err.retryable is False

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(OctomilError) as exc_info:
            raise OctomilError(
                code=OctomilErrorCode.MODEL_NOT_FOUND,
                message="no such model",
            )
        assert exc_info.value.code is OctomilErrorCode.MODEL_NOT_FOUND


# ---------------------------------------------------------------------------
# OctomilError.from_http_status factory
# ---------------------------------------------------------------------------


class TestOctomilErrorFromHttpStatus:
    def test_with_explicit_message(self) -> None:
        err = OctomilError.from_http_status(401, "Invalid token")
        assert err.code is OctomilErrorCode.AUTHENTICATION_FAILED
        assert err.error_message == "Invalid token"

    def test_without_message_uses_default(self) -> None:
        err = OctomilError.from_http_status(503)
        assert err.code is OctomilErrorCode.SERVER_ERROR
        assert err.error_message == "HTTP 503"

    def test_unknown_status_code(self) -> None:
        err = OctomilError.from_http_status(418)
        assert err.code is OctomilErrorCode.UNKNOWN
        assert err.error_message == "HTTP 418"


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestOctomilErrorRepr:
    def test_repr_format(self) -> None:
        err = OctomilError(
            code=OctomilErrorCode.RATE_LIMITED,
            message="slow down",
        )
        r = repr(err)
        assert "code=rate_limited" in r
        assert "retryable=True" in r
        assert "message='slow down'" in r

    def test_repr_non_retryable(self) -> None:
        err = OctomilError(
            code=OctomilErrorCode.FORBIDDEN,
            message="nope",
        )
        r = repr(err)
        assert "retryable=False" in r


# ---------------------------------------------------------------------------
# retry_after_ms — new field (Pillar 3 / Pillar 3-step-3)
# ---------------------------------------------------------------------------


class TestRetryAfterMs:
    def test_rate_limited_with_retry_after_ms(self) -> None:
        err = OctomilError(
            code=OctomilErrorCode.RATE_LIMITED,
            message="x",
            retry_after_ms=1500,
        )
        assert err.retry_after_ms == 1500

    def test_rate_limited_is_retryable(self) -> None:
        err = OctomilError(
            code=OctomilErrorCode.RATE_LIMITED,
            message="x",
            retry_after_ms=1500,
        )
        assert err.retryable is True

    def test_retry_after_ms_defaults_to_none(self) -> None:
        err = OctomilError(
            code=OctomilErrorCode.SERVER_ERROR,
            message="500",
        )
        assert err.retry_after_ms is None

    def test_from_http_status_429_with_retry_after_ms(self) -> None:
        err = OctomilError.from_http_status(429, "Too Many Requests", retry_after_ms=30000)
        assert err.code is OctomilErrorCode.RATE_LIMITED
        assert err.retry_after_ms == 30000
        assert err.retryable is True

    def test_from_http_status_no_retry_after_ms(self) -> None:
        err = OctomilError.from_http_status(500)
        assert err.retry_after_ms is None

    def test_retry_after_ms_zero_is_valid(self) -> None:
        """Zero is a legal hint meaning 'retry immediately'."""
        err = OctomilError(
            code=OctomilErrorCode.RATE_LIMITED,
            message="x",
            retry_after_ms=0,
        )
        assert err.retry_after_ms == 0


# ---------------------------------------------------------------------------
# Newly-added codes from v1.25.0 catalog — importable + correct retryable
# ---------------------------------------------------------------------------


class TestNewV125Codes:
    """Verify all 26 new codes added in the v1.25.0 catalog bump are
    importable via OctomilErrorCode and have the expected .retryable value."""

    @pytest.mark.parametrize(
        ("value", "expected_retryable"),
        [
            # auth category — never retryable
            ("insufficient_scope", False),
            ("missing_org_context", False),
            # catalog category — never retryable
            ("no_default_model", False),
            ("capability_not_supported", False),
            ("previous_response_not_found", False),
            ("app_not_found", False),
            ("capability_not_configured", False),
            ("app_context_conflict", False),
            ("invalid_model_ref", False),
            # runtime category
            ("provider_error", False),  # never
            ("upstream_provider_error", True),  # backoff_safe
            ("too_many_tools", False),  # never
            ("unsupported_tool_calling", False),  # never
            # policy category — never retryable
            ("cloud_inference_not_allowed", False),
            ("hosted_tts_disabled", False),
            ("plan_limit_exceeded", False),
            # control category — never retryable
            ("incident_not_found", False),
            ("deployment_not_found", False),
            ("experiment_not_found", False),
            ("experiment_state_invalid", False),
            # new auth-adjacent resource codes — never retryable
            ("api_key_not_found", False),
            ("api_key_already_revoked", False),
            ("integration_not_found", False),
            ("billing_customer_not_found", False),
            ("action_not_found", False),
            ("action_state_invalid", False),
        ],
    )
    def test_code_importable_and_retryable(self, value: str, expected_retryable: bool) -> None:
        code = OctomilErrorCode(value)
        err = OctomilError(code=code, message="test")
        assert err.retryable is expected_retryable
