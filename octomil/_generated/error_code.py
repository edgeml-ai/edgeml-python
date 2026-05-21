"""Auto-generated from octomil-contracts. Do not edit."""

from enum import Enum
from typing import NamedTuple


class ErrorCode(str, Enum):
    INVALID_API_KEY = "invalid_api_key"
    """401 — API key invalid or missing"""
    AUTHENTICATION_FAILED = "authentication_failed"
    """Auth failure (token expired, revoked, malformed)"""
    FORBIDDEN = "forbidden"
    """403 — insufficient permissions"""
    INSUFFICIENT_SCOPE = "insufficient_scope"
    """API key is valid but lacks the scope required for this operation"""
    MISSING_ORG_CONTEXT = "missing_org_context"
    """Authenticated principal has no associated org_id and the route requires one"""
    DEVICE_NOT_REGISTERED = "device_not_registered"
    """Device has not completed control.register()"""
    TOKEN_EXPIRED = "token_expired"
    """Access token has expired and must be refreshed or reissued"""
    DEVICE_REVOKED = "device_revoked"
    """Device registration has been revoked by an administrator"""
    PASSKEY_CHALLENGE_EXPIRED = "passkey_challenge_expired"
    """400 — WebAuthn challenge has expired or does not exist; start the options step again"""
    PASSKEY_CREDENTIAL_NOT_FOUND = "passkey_credential_not_found"
    """404 — passkey credential is not registered for this user or does not exist"""
    INVALID_TOKEN = "invalid_token"
    """400 — token (reset, verification, transfer) is invalid, malformed, or expired"""
    EMAIL_ALREADY_VERIFIED = "email_already_verified"
    """409 — the user's email is already verified; the requested operation requires an unverified email"""
    EMAIL_ALREADY_IN_USE = "email_already_in_use"
    """409 — the requested email address is already associated with another account"""
    LAST_AUTH_METHOD = "last_auth_method"
    """400 — cannot remove the user's only authentication method; add another method first"""
    OAUTH_PROVIDER_NOT_LINKED = "oauth_provider_not_linked"
    """404 — the specified OAuth provider is not linked to this account"""
    NETWORK_UNAVAILABLE = "network_unavailable"
    """No connectivity"""
    REQUEST_TIMEOUT = "request_timeout"
    """Server did not respond in time"""
    SERVER_ERROR = "server_error"
    """5xx from server"""
    RATE_LIMITED = "rate_limited"
    """429 — too many requests"""
    INVALID_INPUT = "invalid_input"
    """Bad input data (malformed, wrong type, out of range)"""
    UNSUPPORTED_MODALITY = "unsupported_modality"
    """Input modality not supported by the target model"""
    CONTEXT_TOO_LARGE = "context_too_large"
    """Input exceeds model context window"""
    MODEL_NOT_FOUND = "model_not_found"
    """404 — requested model does not exist"""
    NO_DEFAULT_MODEL = "no_default_model"
    """Route requires a default model and the org has not configured one"""
    CAPABILITY_NOT_SUPPORTED = "capability_not_supported"
    """Provider/model does not support the requested capability (e.g. TTS on a chat-only model)"""
    PREVIOUS_RESPONSE_NOT_FOUND = "previous_response_not_found"
    """404 — `previous_response_id` references a response that does not exist or has expired"""
    APP_NOT_FOUND = "app_not_found"
    """404 — referenced app slug does not exist in the caller's organization"""
    CAPABILITY_NOT_CONFIGURED = "capability_not_configured"
    """404 — the referenced app exists but has no slot configured for the requested capability. Distinct from `capability_not_supported` (model-level)."""
    APP_CONTEXT_CONFLICT = "app_context_conflict"
    """403 — `@app/<slug>/<capability>` model ref resolved to a different app than the auth context (API key or X-Octomil-App-Id header)"""
    INVALID_MODEL_REF = "invalid_model_ref"
    """400 — model ref string is malformed (cannot be parsed as a raw model ID or `@app/<slug>/<capability>` ref)"""
    MODEL_DISABLED = "model_disabled"
    """Kill switch active for this model"""
    VERSION_NOT_FOUND = "version_not_found"
    """Requested version does not exist for this model"""
    DOWNLOAD_FAILED = "download_failed"
    """Model download error (network, server, storage)"""
    CHECKSUM_MISMATCH = "checksum_mismatch"
    """Integrity check failed after download"""
    INSUFFICIENT_STORAGE = "insufficient_storage"
    """Not enough disk space for model"""
    INSUFFICIENT_MEMORY = "insufficient_memory"
    """OOM during inference or model loading"""
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    """No compatible runtime for this model format"""
    ACCELERATOR_UNAVAILABLE = "accelerator_unavailable"
    """Required accelerator (GPU, NPU, ANE) not available"""
    MODEL_LOAD_FAILED = "model_load_failed"
    """Runtime initialization error"""
    INFERENCE_FAILED = "inference_failed"
    """Prediction error during inference"""
    PROVIDER_ERROR = "provider_error"
    """Generic non-retryable error from a downstream provider (validation, malformed response, etc.). Distinct from `upstream_provider_error` which is transport-layer."""
    UPSTREAM_PROVIDER_ERROR = "upstream_provider_error"
    """Upstream provider returned a transport-level failure (5xx, timeout, connection reset). Distinct from `provider_error` which is the provider's own application error."""
    TOO_MANY_TOOLS = "too_many_tools"
    """Request supplied more tool definitions than the runtime permits for tool-calling"""
    UNSUPPORTED_TOOL_CALLING = "unsupported_tool_calling"
    """Target model does not support tool/function calling but the request supplied tools"""
    STREAM_INTERRUPTED = "stream_interrupted"
    """Streaming response was interrupted before completion"""
    POLICY_DENIED = "policy_denied"
    """Routing policy explicitly denied the request"""
    CLOUD_FALLBACK_DISALLOWED = "cloud_fallback_disallowed"
    """Local inference failed and cloud fallback is disabled by policy"""
    CLOUD_INFERENCE_NOT_ALLOWED = "cloud_inference_not_allowed"
    """Cloud inference is disabled for this principal (scope or org-level policy).  Distinct from `cloud_fallback_disallowed` which fires only after a local attempt failed."""
    HOSTED_TTS_DISABLED = "hosted_tts_disabled"
    """Hosted TTS is disabled at the deployment level (operator-controlled kill switch)"""
    PLAN_LIMIT_EXCEEDED = "plan_limit_exceeded"
    """Request exceeds the plan's quota or rate limit and admission was rejected"""
    CLOUD_CREDENTIALS_MISSING = "cloud_credentials_missing"
    """No provider credential configured for the requested cloud provider (no org BYOK and no Octomil-managed credential available)"""
    CLOUD_CREDENTIALS_REVOKED = "cloud_credentials_revoked"
    """The provider credential used for cloud inference has been revoked"""
    CLOUD_PROVIDER_AUTH_FAILED = "cloud_provider_auth_failed"
    """Cloud provider rejected the credential (invalid API key, expired, or insufficient permissions)"""
    MAX_TOOL_ROUNDS_EXCEEDED = "max_tool_rounds_exceeded"
    """Tool execution loop hit the iteration limit"""
    TRAINING_FAILED = "training_failed"
    """Training, aggregation, or update-processing operation failed"""
    TRAINING_NOT_SUPPORTED = "training_not_supported"
    """Requested training/update flow is not supported for this model, runtime, platform, or workspace"""
    WEIGHT_UPLOAD_FAILED = "weight_upload_failed"
    """Uploading model weights, deltas, or training updates failed before successful server acceptance"""
    CONTROL_SYNC_FAILED = "control_sync_failed"
    """Control plane sync returned an error"""
    ASSIGNMENT_NOT_FOUND = "assignment_not_found"
    """No model assignment exists for this device/experiment"""
    INCIDENT_NOT_FOUND = "incident_not_found"
    """Incident does not exist or has been deleted"""
    DEPLOYMENT_NOT_FOUND = "deployment_not_found"
    """Deployment does not exist or has been deleted"""
    EXPERIMENT_NOT_FOUND = "experiment_not_found"
    """Experiment does not exist or has been deleted"""
    EXPERIMENT_STATE_INVALID = "experiment_state_invalid"
    """400 — requested experiment lifecycle transition (start/pause/resume/complete/cancel) is not valid from the current status"""
    API_KEY_NOT_FOUND = "api_key_not_found"
    """404 — API key with the given key_id does not exist or belongs to a different org"""
    API_KEY_ALREADY_REVOKED = "api_key_already_revoked"
    """400 — attempted to rotate or revoke an API key that is already revoked"""
    INTEGRATION_NOT_FOUND = "integration_not_found"
    """404 — integration with the given integration_id does not exist in this org"""
    BILLING_CUSTOMER_NOT_FOUND = "billing_customer_not_found"
    """404 — no Stripe billing customer exists for this org; complete checkout first"""
    ACTION_NOT_FOUND = "action_not_found"
    """404 — operations action does not exist for this deployment"""
    ACTION_STATE_INVALID = "action_state_invalid"
    """409 — the action is not in `proposed` status and cannot be executed or dismissed"""
    CREDENTIAL_NOT_FOUND = "credential_not_found"
    """404 — BYOK cloud credential with the given credential_id does not exist in this org"""
    CONNECTION_NOT_FOUND = "connection_not_found"
    """404 — cloud provider connection with the given connection_id does not exist in this org"""
    LOCAL_RUNTIME_NOT_FOUND = "local_runtime_not_found"
    """404 — local runtime target with the given runtime_id does not exist in this org"""
    CHECKOUT_NOT_COMPLETE = "checkout_not_complete"
    """409 — Stripe Checkout session has not yet completed (no subscription attached)"""
    UPSTREAM_PROVIDER_UNAVAILABLE = "upstream_provider_unavailable"
    """502 — upstream cloud provider returned an unexpected error or could not be reached during connection test"""
    AGENT_SYSTEM_UNAVAILABLE = "agent_system_unavailable"
    """503 — the agent system (registry or model client) has not initialized yet or failed to start. Caller should retry after a short delay.
"""
    THREAD_NOT_FOUND = "thread_not_found"
    """404 — agent thread does not exist or belongs to a different org"""
    RUN_NOT_FOUND = "run_not_found"
    """404 — agent run does not exist or belongs to a different org"""
    RUN_STATE_INVALID = "run_state_invalid"
    """409 — the requested operation (cancel, respond) is not valid from the run's current status
"""
    APPROVAL_NOT_FOUND = "approval_not_found"
    """404 — approval request does not exist or the run belongs to a different org"""
    APPROVAL_ALREADY_RESOLVED = "approval_already_resolved"
    """409 — approval request has already been resolved (approved or rejected)"""
    JOB_NOT_FOUND = "job_not_found"
    """404 — job does not exist"""
    JOB_STATE_INVALID = "job_state_invalid"
    """400 — the requested operation (cancel, retry) is not valid from the job's current status
"""
    CANCELLED = "cancelled"
    """User or caller cancelled the operation"""
    APP_BACKGROUNDED = "app_backgrounded"
    """App moved to background, operation stopped"""
    UNKNOWN = "unknown"
    """Catch-all for unrecognized errors. SDKs MUST map unrecognized codes here."""


class ErrorCategory(str, Enum):
    AUTH = "auth"
    """Auth / Access"""
    NETWORK = "network"
    """Network / Transport"""
    INPUT = "input"
    """Input / Validation"""
    CATALOG = "catalog"
    """Catalog / Model Resolution"""
    DOWNLOAD = "download"
    """Download / Artifact Integrity"""
    DEVICE = "device"
    """Device / Environment"""
    RUNTIME = "runtime"
    """Runtime / Inference"""
    POLICY = "policy"
    """Policy / Routing"""
    TRAINING = "training"
    """Training / Federated Learning"""
    CONTROL = "control"
    """Control Plane / Rollout"""
    LIFECYCLE = "lifecycle"
    """Cancellation / Lifecycle"""
    UNKNOWN = "unknown"
    """Unknown"""


class RetryClass(str, Enum):
    NEVER = "never"
    IMMEDIATE_SAFE = "immediate_safe"
    BACKOFF_SAFE = "backoff_safe"
    CONDITIONAL = "conditional"


class SuggestedAction(str, Enum):
    FIX_CREDENTIALS = "fix_credentials"
    REAUTHENTICATE = "reauthenticate"
    CHECK_PERMISSIONS = "check_permissions"
    REGISTER_DEVICE = "register_device"
    FIX_REQUEST = "fix_request"
    RETRY_OR_FALLBACK = "retry_or_fallback"
    RETRY = "retry"
    RETRY_AFTER = "retry_after"
    REDUCE_INPUT_OR_FALLBACK = "reduce_input_or_fallback"
    CHECK_MODEL_ID = "check_model_id"
    USE_ALTERNATE_MODEL = "use_alternate_model"
    CHECK_VERSION = "check_version"
    REDOWNLOAD = "redownload"
    FREE_STORAGE_OR_FALLBACK = "free_storage_or_fallback"
    TRY_SMALLER_MODEL = "try_smaller_model"
    TRY_ALTERNATE_RUNTIME = "try_alternate_runtime"
    TRY_CPU_OR_FALLBACK = "try_cpu_or_fallback"
    CHECK_POLICY = "check_policy"
    CHANGE_POLICY_OR_FIX_LOCAL = "change_policy_or_fix_local"
    INCREASE_LIMIT_OR_SIMPLIFY = "increase_limit_or_simplify"
    CHECK_ASSIGNMENT = "check_assignment"
    NONE = "none"
    RESUME_ON_FOREGROUND = "resume_on_foreground"
    REPORT_BUG = "report_bug"


class ErrorClassification(NamedTuple):
    category: ErrorCategory
    retry_class: RetryClass
    fallback_eligible: bool
    suggested_action: SuggestedAction


ERROR_CLASSIFICATION: dict[ErrorCode, ErrorClassification] = {
    ErrorCode.INVALID_API_KEY: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.FIX_CREDENTIALS
    ),
    ErrorCode.AUTHENTICATION_FAILED: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.REAUTHENTICATE
    ),
    ErrorCode.FORBIDDEN: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.CHECK_PERMISSIONS
    ),
    ErrorCode.INSUFFICIENT_SCOPE: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.CHECK_PERMISSIONS
    ),
    ErrorCode.MISSING_ORG_CONTEXT: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.CHECK_PERMISSIONS
    ),
    ErrorCode.DEVICE_NOT_REGISTERED: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.REGISTER_DEVICE
    ),
    ErrorCode.TOKEN_EXPIRED: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.REAUTHENTICATE
    ),
    ErrorCode.DEVICE_REVOKED: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.REGISTER_DEVICE
    ),
    ErrorCode.PASSKEY_CHALLENGE_EXPIRED: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.REAUTHENTICATE
    ),
    ErrorCode.PASSKEY_CREDENTIAL_NOT_FOUND: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.INVALID_TOKEN: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.EMAIL_ALREADY_VERIFIED: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.EMAIL_ALREADY_IN_USE: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.LAST_AUTH_METHOD: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.OAUTH_PROVIDER_NOT_LINKED: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.NETWORK_UNAVAILABLE: ErrorClassification(
        ErrorCategory.NETWORK, RetryClass.BACKOFF_SAFE, True, SuggestedAction.RETRY_OR_FALLBACK
    ),
    ErrorCode.REQUEST_TIMEOUT: ErrorClassification(
        ErrorCategory.NETWORK, RetryClass.CONDITIONAL, True, SuggestedAction.RETRY_OR_FALLBACK
    ),
    ErrorCode.SERVER_ERROR: ErrorClassification(
        ErrorCategory.NETWORK, RetryClass.BACKOFF_SAFE, True, SuggestedAction.RETRY
    ),
    ErrorCode.RATE_LIMITED: ErrorClassification(
        ErrorCategory.NETWORK, RetryClass.CONDITIONAL, False, SuggestedAction.RETRY_AFTER
    ),
    ErrorCode.INVALID_INPUT: ErrorClassification(
        ErrorCategory.INPUT, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.UNSUPPORTED_MODALITY: ErrorClassification(
        ErrorCategory.INPUT, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.CONTEXT_TOO_LARGE: ErrorClassification(
        ErrorCategory.INPUT, RetryClass.NEVER, True, SuggestedAction.REDUCE_INPUT_OR_FALLBACK
    ),
    ErrorCode.MODEL_NOT_FOUND: ErrorClassification(
        ErrorCategory.CATALOG, RetryClass.NEVER, False, SuggestedAction.CHECK_MODEL_ID
    ),
    ErrorCode.NO_DEFAULT_MODEL: ErrorClassification(
        ErrorCategory.CATALOG, RetryClass.NEVER, False, SuggestedAction.CHECK_MODEL_ID
    ),
    ErrorCode.CAPABILITY_NOT_SUPPORTED: ErrorClassification(
        ErrorCategory.CATALOG, RetryClass.NEVER, True, SuggestedAction.USE_ALTERNATE_MODEL
    ),
    ErrorCode.PREVIOUS_RESPONSE_NOT_FOUND: ErrorClassification(
        ErrorCategory.CATALOG, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.APP_NOT_FOUND: ErrorClassification(
        ErrorCategory.CATALOG, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.CAPABILITY_NOT_CONFIGURED: ErrorClassification(
        ErrorCategory.CATALOG, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.APP_CONTEXT_CONFLICT: ErrorClassification(
        ErrorCategory.CATALOG, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.INVALID_MODEL_REF: ErrorClassification(
        ErrorCategory.CATALOG, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.MODEL_DISABLED: ErrorClassification(
        ErrorCategory.CATALOG, RetryClass.NEVER, True, SuggestedAction.USE_ALTERNATE_MODEL
    ),
    ErrorCode.VERSION_NOT_FOUND: ErrorClassification(
        ErrorCategory.CATALOG, RetryClass.NEVER, False, SuggestedAction.CHECK_VERSION
    ),
    ErrorCode.DOWNLOAD_FAILED: ErrorClassification(
        ErrorCategory.DOWNLOAD, RetryClass.BACKOFF_SAFE, True, SuggestedAction.RETRY_OR_FALLBACK
    ),
    ErrorCode.CHECKSUM_MISMATCH: ErrorClassification(
        ErrorCategory.DOWNLOAD, RetryClass.CONDITIONAL, False, SuggestedAction.REDOWNLOAD
    ),
    ErrorCode.INSUFFICIENT_STORAGE: ErrorClassification(
        ErrorCategory.DEVICE, RetryClass.NEVER, True, SuggestedAction.FREE_STORAGE_OR_FALLBACK
    ),
    ErrorCode.INSUFFICIENT_MEMORY: ErrorClassification(
        ErrorCategory.DEVICE, RetryClass.NEVER, True, SuggestedAction.TRY_SMALLER_MODEL
    ),
    ErrorCode.RUNTIME_UNAVAILABLE: ErrorClassification(
        ErrorCategory.DEVICE, RetryClass.NEVER, True, SuggestedAction.TRY_ALTERNATE_RUNTIME
    ),
    ErrorCode.ACCELERATOR_UNAVAILABLE: ErrorClassification(
        ErrorCategory.DEVICE, RetryClass.NEVER, True, SuggestedAction.TRY_CPU_OR_FALLBACK
    ),
    ErrorCode.MODEL_LOAD_FAILED: ErrorClassification(
        ErrorCategory.RUNTIME, RetryClass.CONDITIONAL, True, SuggestedAction.RETRY_OR_FALLBACK
    ),
    ErrorCode.INFERENCE_FAILED: ErrorClassification(
        ErrorCategory.RUNTIME, RetryClass.CONDITIONAL, True, SuggestedAction.RETRY_OR_FALLBACK
    ),
    ErrorCode.PROVIDER_ERROR: ErrorClassification(
        ErrorCategory.RUNTIME, RetryClass.NEVER, True, SuggestedAction.RETRY_OR_FALLBACK
    ),
    ErrorCode.UPSTREAM_PROVIDER_ERROR: ErrorClassification(
        ErrorCategory.RUNTIME, RetryClass.BACKOFF_SAFE, True, SuggestedAction.RETRY_OR_FALLBACK
    ),
    ErrorCode.TOO_MANY_TOOLS: ErrorClassification(
        ErrorCategory.RUNTIME, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.UNSUPPORTED_TOOL_CALLING: ErrorClassification(
        ErrorCategory.RUNTIME, RetryClass.NEVER, True, SuggestedAction.USE_ALTERNATE_MODEL
    ),
    ErrorCode.STREAM_INTERRUPTED: ErrorClassification(
        ErrorCategory.RUNTIME, RetryClass.IMMEDIATE_SAFE, True, SuggestedAction.RETRY
    ),
    ErrorCode.POLICY_DENIED: ErrorClassification(
        ErrorCategory.POLICY, RetryClass.NEVER, False, SuggestedAction.CHECK_POLICY
    ),
    ErrorCode.CLOUD_FALLBACK_DISALLOWED: ErrorClassification(
        ErrorCategory.POLICY, RetryClass.NEVER, False, SuggestedAction.CHANGE_POLICY_OR_FIX_LOCAL
    ),
    ErrorCode.CLOUD_INFERENCE_NOT_ALLOWED: ErrorClassification(
        ErrorCategory.POLICY, RetryClass.NEVER, False, SuggestedAction.CHECK_POLICY
    ),
    ErrorCode.HOSTED_TTS_DISABLED: ErrorClassification(
        ErrorCategory.POLICY, RetryClass.NEVER, False, SuggestedAction.CHECK_POLICY
    ),
    ErrorCode.PLAN_LIMIT_EXCEEDED: ErrorClassification(
        ErrorCategory.POLICY, RetryClass.NEVER, False, SuggestedAction.INCREASE_LIMIT_OR_SIMPLIFY
    ),
    ErrorCode.CLOUD_CREDENTIALS_MISSING: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.FIX_CREDENTIALS
    ),
    ErrorCode.CLOUD_CREDENTIALS_REVOKED: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.FIX_CREDENTIALS
    ),
    ErrorCode.CLOUD_PROVIDER_AUTH_FAILED: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.FIX_CREDENTIALS
    ),
    ErrorCode.MAX_TOOL_ROUNDS_EXCEEDED: ErrorClassification(
        ErrorCategory.POLICY, RetryClass.NEVER, False, SuggestedAction.INCREASE_LIMIT_OR_SIMPLIFY
    ),
    ErrorCode.TRAINING_FAILED: ErrorClassification(
        ErrorCategory.TRAINING, RetryClass.CONDITIONAL, False, SuggestedAction.RETRY
    ),
    ErrorCode.TRAINING_NOT_SUPPORTED: ErrorClassification(
        ErrorCategory.TRAINING, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.WEIGHT_UPLOAD_FAILED: ErrorClassification(
        ErrorCategory.TRAINING, RetryClass.BACKOFF_SAFE, False, SuggestedAction.RETRY
    ),
    ErrorCode.CONTROL_SYNC_FAILED: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.BACKOFF_SAFE, False, SuggestedAction.RETRY
    ),
    ErrorCode.ASSIGNMENT_NOT_FOUND: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.NEVER, False, SuggestedAction.CHECK_ASSIGNMENT
    ),
    ErrorCode.INCIDENT_NOT_FOUND: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.NEVER, False, SuggestedAction.NONE
    ),
    ErrorCode.DEPLOYMENT_NOT_FOUND: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.EXPERIMENT_NOT_FOUND: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.EXPERIMENT_STATE_INVALID: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.API_KEY_NOT_FOUND: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.API_KEY_ALREADY_REVOKED: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.INTEGRATION_NOT_FOUND: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.BILLING_CUSTOMER_NOT_FOUND: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.ACTION_NOT_FOUND: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.ACTION_STATE_INVALID: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.CREDENTIAL_NOT_FOUND: ErrorClassification(
        ErrorCategory.AUTH, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.CONNECTION_NOT_FOUND: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.LOCAL_RUNTIME_NOT_FOUND: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.CHECKOUT_NOT_COMPLETE: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.CONDITIONAL, False, SuggestedAction.RETRY
    ),
    ErrorCode.UPSTREAM_PROVIDER_UNAVAILABLE: ErrorClassification(
        ErrorCategory.NETWORK, RetryClass.BACKOFF_SAFE, False, SuggestedAction.RETRY
    ),
    ErrorCode.AGENT_SYSTEM_UNAVAILABLE: ErrorClassification(
        ErrorCategory.NETWORK, RetryClass.BACKOFF_SAFE, False, SuggestedAction.RETRY
    ),
    ErrorCode.THREAD_NOT_FOUND: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.RUN_NOT_FOUND: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.RUN_STATE_INVALID: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.APPROVAL_NOT_FOUND: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.APPROVAL_ALREADY_RESOLVED: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.JOB_NOT_FOUND: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.JOB_STATE_INVALID: ErrorClassification(
        ErrorCategory.CONTROL, RetryClass.NEVER, False, SuggestedAction.FIX_REQUEST
    ),
    ErrorCode.CANCELLED: ErrorClassification(ErrorCategory.LIFECYCLE, RetryClass.NEVER, False, SuggestedAction.NONE),
    ErrorCode.APP_BACKGROUNDED: ErrorClassification(
        ErrorCategory.LIFECYCLE, RetryClass.CONDITIONAL, False, SuggestedAction.RESUME_ON_FOREGROUND
    ),
    ErrorCode.UNKNOWN: ErrorClassification(ErrorCategory.UNKNOWN, RetryClass.NEVER, False, SuggestedAction.REPORT_BUG),
}
