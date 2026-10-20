"""Google Cloud Secret Manager integration with TTL caching and secure environment fallback."""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from thermofisher_agent.observability import structured_logger, trace_span
from thermofisher_agent.schemas import Field, StrictToolOutputModel

try:
    from google.cloud import secretmanager  # type: ignore[import-untyped]

    HAS_GCP_SECRET_MANAGER = True
except ImportError:
    HAS_GCP_SECRET_MANAGER = False
    secretmanager = None


class SecretResolutionMetadata(StrictToolOutputModel):
    """Audit metadata describing how a secret was resolved (never contains the raw secret value)."""

    secret_id: str = Field(..., description="Logical secret identifier.")
    source: str = Field(
        ...,
        description="Resolution source ('gcp_secret_manager', 'ttl_cache', 'env_fallback', or 'not_configured').",
    )
    resource_name: Optional[str] = Field(
        default=None,
        description="Full Google Cloud Secret Manager resource path (`projects/*/secrets/*/versions/*`).",
    )
    is_configured: bool = Field(..., description="True if a valid non-placeholder secret was resolved.")


class SecretManagerService:
    """Dedicated Google Cloud Secret Manager client with TTL caching and rotation support."""

    def __init__(
        self,
        project_id: str | None = None,
        *,
        cache_ttl_seconds: float = 300.0,
        client: Any | None = None,
    ) -> None:
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.cache_ttl_seconds = cache_ttl_seconds
        self._injected_client = client
        self._cache: dict[str, tuple[str, float, str]] = {}

    def _build_resource_name(self, secret_id: str, version: str = "latest") -> str:
        proj = self.project_id or "thermofisher-local-project"
        return f"projects/{proj}/secrets/{secret_id}/versions/{version}"

    def invalidate_cache(self, secret_id: str | None = None) -> None:
        """Invalidates cached secrets to support zero-downtime key rotation."""
        if secret_id is None:
            self._cache.clear()
        else:
            self._cache.pop(secret_id, None)

    @trace_span("thermofisher.secrets.get_secret", component="secrets")
    def get_secret(
        self,
        secret_id: str,
        *,
        env_var_fallback: str | None = None,
        version: str = "latest",
    ) -> tuple[str | None, SecretResolutionMetadata]:
        """Retrieves a secret from Google Cloud Secret Manager with TTL caching and local dev fallback."""
        now = time.monotonic()
        if secret_id in self._cache:
            cached_val, expires_at, resource_name = self._cache[secret_id]
            if now < expires_at:
                return cached_val, SecretResolutionMetadata(
                    secret_id=secret_id,
                    source="ttl_cache",
                    resource_name=resource_name,
                    is_configured=True,
                )

        resource_name = self._build_resource_name(secret_id, version=version)
        sm_client = self._injected_client
        if sm_client is None and HAS_GCP_SECRET_MANAGER and self.project_id and secretmanager is not None:
            try:
                sm_client = secretmanager.SecretManagerServiceClient()
            except Exception as exc:
                structured_logger.warning(
                    f"Could not initialize GCP SecretManagerServiceClient: {exc}",
                    component="secrets",
                    event_type="secret_manager_init_fallback",
                )
                sm_client = None

        if sm_client is not None:
            try:
                response = sm_client.access_secret_version(request={"name": resource_name})
                payload_bytes = response.payload.data
                secret_value = (
                    payload_bytes.decode("utf-8").strip()
                    if isinstance(payload_bytes, (bytes, bytearray))
                    else str(payload_bytes).strip()
                )
                if secret_value and secret_value != "your_gemini_api_key_here":
                    self._cache[secret_id] = (secret_value, now + self.cache_ttl_seconds, resource_name)
                    structured_logger.info(
                        f"Resolved secret '{secret_id}' from Google Cloud Secret Manager",
                        component="secrets",
                        event_type="secret_resolved_gcp",
                        attributes={"secret_id": secret_id, "resource_name": resource_name},
                    )
                    return secret_value, SecretResolutionMetadata(
                        secret_id=secret_id,
                        source="gcp_secret_manager",
                        resource_name=resource_name,
                        is_configured=True,
                    )
            except Exception as exc:
                structured_logger.warning(
                    f"GCP Secret Manager lookup failed for '{resource_name}', falling back to environment: {exc}",
                    component="secrets",
                    event_type="secret_manager_lookup_fallback",
                    attributes={"secret_id": secret_id, "resource_name": resource_name},
                )

        # Fallback to environment variables for local development / CI
        candidates = [
            env_var_fallback,
            secret_id.upper().replace("-", "_"),
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
        ]
        for env_key in candidates:
            if not env_key:
                continue
            val = os.getenv(env_key, "").strip()
            if val and val != "your_gemini_api_key_here":
                self._cache[secret_id] = (val, now + self.cache_ttl_seconds, f"env://{env_key}")
                return val, SecretResolutionMetadata(
                    secret_id=secret_id,
                    source="env_fallback",
                    resource_name=f"env://{env_key}",
                    is_configured=True,
                )

        return None, SecretResolutionMetadata(
            secret_id=secret_id,
            source="not_configured",
            resource_name=resource_name,
            is_configured=False,
        )

    def get_gemini_api_key(self) -> tuple[str | None, SecretResolutionMetadata]:
        """Resolves the Gemini API key via Google Cloud Secret Manager (`thermofisher-gemini-api-key`)."""
        secret_name = os.getenv("THERMOFISHER_GEMINI_SECRET_ID", "thermofisher-gemini-api-key")
        return self.get_secret(secret_name, env_var_fallback="GEMINI_API_KEY")


secret_manager_service = SecretManagerService()

__all__ = [
    "HAS_GCP_SECRET_MANAGER",
    "SecretManagerService",
    "SecretResolutionMetadata",
    "secret_manager_service",
]
