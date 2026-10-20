"""Observability, OpenTelemetry Distributed Tracing, Structured JSON Logging, PII Redaction, and Intent-vs-Outcome Tracking."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import json
import logging
import os
import re
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Literal, Optional

from thermofisher_agent.schemas import Field, StrictToolOutputModel

# Optional OpenTelemetry SDK integration for Google Cloud Trace / OTLP
try:
    from opentelemetry import trace as otel_trace  # type: ignore[import-untyped]
    from opentelemetry.trace import Status as OtelStatus, StatusCode as OtelStatusCode  # type: ignore[import-untyped]

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False
    otel_trace = None
    OtelStatus = None
    OtelStatusCode = None


# Context variables for distributed trace propagation across sync and async call chains
_CURRENT_TRACE_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("tf_trace_id", default=None)
_CURRENT_SPAN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("tf_span_id", default=None)
_CURRENT_SESSION_ID: contextvars.ContextVar[str] = contextvars.ContextVar("tf_session_id", default="default")


# ============================================================================
# 1. PII Redaction Engine
# ============================================================================


class PIIRedactionResult(StrictToolOutputModel):
    """Structured result of PII scrubbing on user or agent text."""

    redacted_text: str = Field(..., description="Text with all detected PII tokens replaced by placeholders.")
    redaction_count: int = Field(default=0, ge=0, description="Total number of PII entities redacted.")
    detected_pii_types: list[str] = Field(
        default_factory=list,
        description="Categories of PII found (e.g., EMAIL, PHONE, CREDIT_CARD, PASSPORT, GOV_ID, IP_ADDRESS).",
    )


class PIIRedactor:
    """Scrubs sensitive Personal Identifiable Information (PII) before logging, tracing, or persistence."""

    _PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
        (
            "EMAIL",
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            "[REDACTED_EMAIL]",
        ),
        (
            "CREDIT_CARD",
            re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
            "[REDACTED_CREDIT_CARD]",
        ),
        (
            "PHONE",
            re.compile(
                r"(?:(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]\d{3,4}[\s.-]\d{4})\b"
            ),
            "[REDACTED_PHONE]",
        ),
        (
            "PASSPORT",
            re.compile(r"\b(?:passport(?:\s*(?:no|number|#))?[\s:=]+)([A-Z0-9]{6,9})\b", re.IGNORECASE),
            "passport: [REDACTED_PASSPORT]",
        ),
        (
            "GOV_ID",
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "[REDACTED_GOV_ID]",
        ),
        (
            "IP_ADDRESS",
            re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"),
            "[REDACTED_IP]",
        ),
    ]

    def __init__(self) -> None:
        self.total_redactions: int = 0

    def redact_text(self, text: str) -> PIIRedactionResult:
        """Scans text and replaces all detected PII patterns with safe tokens."""
        if not text:
            return PIIRedactionResult(redacted_text="", redaction_count=0, detected_pii_types=[])

        scrubbed = text
        count = 0
        detected_types: list[str] = []

        for pii_type, pattern, replacement in self._PATTERNS:
            matches = list(pattern.finditer(scrubbed))
            if matches:
                count += len(matches)
                if pii_type not in detected_types:
                    detected_types.append(pii_type)
                scrubbed = pattern.sub(replacement, scrubbed)

        self.total_redactions += count
        return PIIRedactionResult(
            redacted_text=scrubbed,
            redaction_count=count,
            detected_pii_types=detected_types,
        )

    def redact_payload(self, payload: Any) -> Any:
        """Recursively scrubs strings inside dicts, lists, and tuples."""
        if isinstance(payload, str):
            return self.redact_text(payload).redacted_text
        if isinstance(payload, dict):
            return {
                str(k): (
                    "***REDACTED_SECRET***"
                    if any(s in str(k).lower() for s in ("api_key", "secret", "password", "token", "authorization"))
                    else self.redact_payload(v)
                )
                for k, v in payload.items()
            }
        if isinstance(payload, list):
            return [self.redact_payload(item) for item in payload]
        if isinstance(payload, tuple):
            return tuple(self.redact_payload(item) for item in payload)
        return payload


# ============================================================================
# 2. Structured JSON Logger (Google Cloud Logging Compatible)
# ============================================================================


class StructuredLogEntry(StrictToolOutputModel):
    """Schema for a single structured JSON log record."""

    timestamp: str = Field(..., description="ISO-8601 UTC timestamp.")
    severity: str = Field(..., description="Cloud Logging severity (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    message: str = Field(..., description="PII-scrubbed log message.")
    component: str = Field(..., description="Subsystem emitting the log (e.g., tools, memory, router, guardrail).")
    event_type: str = Field(..., description="Machine-readable event identifier.")
    session_id: str = Field(default="default", description="Active conversation session ID.")
    trace_id: Optional[str] = Field(default=None, description="W3C / Cloud Trace 32-char hex trace ID.")
    span_id: Optional[str] = Field(default=None, description="16-char hex span ID.")
    latency_ms: Optional[float] = Field(default=None, description="Operation duration in milliseconds.")
    attributes: dict[str, Any] = Field(default_factory=dict, description="PII-scrubbed structured metadata.")


class StructuredJSONLogger:
    """Emits PII-redacted structured JSON logs compatible with Google Cloud Logging."""

    def __init__(self, service_name: str = "thermofisher-coa-agent", redactor: PIIRedactor | None = None) -> None:
        self.service_name = service_name
        self.redactor = redactor or PIIRedactor()
        self.entries: list[StructuredLogEntry] = []
        self._max_buffer = 500
        self._stdlib_logger = logging.getLogger(service_name)
        if not self._stdlib_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._stdlib_logger.addHandler(handler)
            self._stdlib_logger.setLevel(os.getenv("THERMOFISHER_LOG_LEVEL", "WARNING"))

    def log(
        self,
        severity: str,
        message: str,
        *,
        component: str = "agent",
        event_type: str = "general",
        session_id: str | None = None,
        latency_ms: float | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> StructuredLogEntry:
        """Records a structured JSON log entry with automatic PII redaction and trace correlation."""
        clean_msg = self.redactor.redact_text(message).redacted_text
        clean_attrs = self.redactor.redact_payload(attributes or {})
        trace_id = _CURRENT_TRACE_ID.get()
        span_id = _CURRENT_SPAN_ID.get()
        resolved_session = session_id or _CURRENT_SESSION_ID.get()

        entry = StructuredLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            severity=severity.upper(),
            message=clean_msg,
            component=component,
            event_type=event_type,
            session_id=resolved_session,
            trace_id=trace_id,
            span_id=span_id,
            latency_ms=round(latency_ms, 3) if latency_ms is not None else None,
            attributes=clean_attrs,
        )
        self.entries.append(entry)
        if len(self.entries) > self._max_buffer:
            self.entries = self.entries[-self._max_buffer :]

        project_id = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "thermofisher-local"
        cloud_payload = {
            "severity": entry.severity,
            "timestamp": entry.timestamp,
            "message": entry.message,
            "component": entry.component,
            "event_type": entry.event_type,
            "session_id": entry.session_id,
            "logging.googleapis.com/trace": f"projects/{project_id}/traces/{trace_id}" if trace_id else None,
            "logging.googleapis.com/spanId": span_id,
            "latency_ms": entry.latency_ms,
            "attributes": entry.attributes,
        }
        log_level = getattr(logging, entry.severity, logging.INFO)
        self._stdlib_logger.log(log_level, json.dumps(cloud_payload, ensure_ascii=False))
        return entry

    def info(self, message: str, **kwargs: Any) -> StructuredLogEntry:
        return self.log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> StructuredLogEntry:
        return self.log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> StructuredLogEntry:
        return self.log("ERROR", message, **kwargs)


# ============================================================================
# 3. Distributed Tracing Engine (OpenTelemetry + W3C TraceContext)
# ============================================================================


class TelemetrySpanRecord(StrictToolOutputModel):
    """Schema for an OpenTelemetry-compatible distributed trace span."""

    trace_id: str = Field(..., description="32-character lowercase hex W3C trace ID.")
    span_id: str = Field(..., description="16-character lowercase hex span ID.")
    parent_span_id: Optional[str] = Field(default=None, description="Parent span ID if nested.")
    name: str = Field(..., description="Span operation name (e.g., thermofisher.tool.search_certificates).")
    component: str = Field(default="agent", description="Instrumented component.")
    status: Literal["OK", "ERROR"] = Field(default="OK", description="Span completion status.")
    start_time_utc: str = Field(..., description="ISO-8601 start timestamp.")
    end_time_utc: str = Field(..., description="ISO-8601 end timestamp.")
    duration_ms: float = Field(..., ge=0.0, description="Span execution duration in milliseconds.")
    attributes: dict[str, Any] = Field(default_factory=dict, description="PII-redacted span attributes.")
    events: list[dict[str, Any]] = Field(default_factory=list, description="Span lifecycle events.")
    w3c_traceparent: str = Field(..., description="W3C traceparent header (`00-<trace_id>-<span_id>-01`).")


class ActiveSpanHandle:
    """Mutable handle yielded inside a `start_as_current_span` block."""

    def __init__(
        self,
        name: str,
        component: str,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        redactor: PIIRedactor,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.component = component
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self._redactor = redactor
        self.status: Literal["OK", "ERROR"] = "OK"
        self.attributes: dict[str, Any] = self._redactor.redact_payload(attributes or {})
        self.events: list[dict[str, Any]] = []
        self.start_time_utc = datetime.now(timezone.utc).isoformat()
        self._start_perf = time.perf_counter()

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = self._redactor.redact_payload(value)

    def add_event(self, event_name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append(
            {
                "name": event_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attributes": self._redactor.redact_payload(attributes or {}),
            }
        )

    def set_error(self, exc: BaseException | str) -> None:
        self.status = "ERROR"
        self.set_attribute("error.message", str(exc))
        self.set_attribute("error.type", type(exc).__name__ if isinstance(exc, BaseException) else "Error")

    def finalize(self) -> TelemetrySpanRecord:
        duration_ms = round((time.perf_counter() - self._start_perf) * 1000.0, 3)
        return TelemetrySpanRecord(
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            name=self.name,
            component=self.component,
            status=self.status,
            start_time_utc=self.start_time_utc,
            end_time_utc=datetime.now(timezone.utc).isoformat(),
            duration_ms=duration_ms,
            attributes=self.attributes,
            events=self.events,
            w3c_traceparent=f"00-{self.trace_id}-{self.span_id}-01",
        )


class DistributedTracer:
    """OpenTelemetry-compatible distributed tracer with parent-child span correlation."""

    def __init__(self, redactor: PIIRedactor | None = None, logger: StructuredJSONLogger | None = None) -> None:
        self.redactor = redactor or PIIRedactor()
        self.logger = logger
        self.completed_spans: list[TelemetrySpanRecord] = []
        self._max_spans = 500
        self._otel_tracer = otel_trace.get_tracer("thermofisher_agent") if HAS_OPENTELEMETRY and otel_trace else None

    @contextlib.contextmanager
    def start_as_current_span(
        self,
        name: str,
        *,
        component: str = "agent",
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[ActiveSpanHandle]:
        """Starts a distributed trace span, propagating trace_id and parent_span_id via contextvars."""
        parent_trace_id = _CURRENT_TRACE_ID.get()
        parent_span_id = _CURRENT_SPAN_ID.get()
        trace_id = parent_trace_id or secrets.token_hex(16)
        span_id = secrets.token_hex(8)

        token_trace = _CURRENT_TRACE_ID.set(trace_id)
        token_span = _CURRENT_SPAN_ID.set(span_id)

        handle = ActiveSpanHandle(
            name=name,
            component=component,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            redactor=self.redactor,
            attributes=attributes,
        )

        otel_ctx = (
            self._otel_tracer.start_as_current_span(name)
            if self._otel_tracer is not None
            else contextlib.nullcontext()
        )

        with otel_ctx as otel_span:
            try:
                yield handle
            except Exception as exc:
                handle.set_error(exc)
                if otel_span is not None and HAS_OPENTELEMETRY and OtelStatus and OtelStatusCode:
                    otel_span.set_status(OtelStatus(OtelStatusCode.ERROR, str(exc)))
                raise
            finally:
                record = handle.finalize()
                self.completed_spans.append(record)
                if len(self.completed_spans) > self._max_spans:
                    self.completed_spans = self.completed_spans[-self._max_spans :]
                if self.logger is not None:
                    self.logger.log(
                        "ERROR" if record.status == "ERROR" else "INFO",
                        f"Span completed: {record.name} ({record.duration_ms}ms)",
                        component=component,
                        event_type="span_complete",
                        latency_ms=record.duration_ms,
                        attributes={"span_name": record.name, "status": record.status, **record.attributes},
                    )
                _CURRENT_SPAN_ID.reset(token_span)
                _CURRENT_TRACE_ID.reset(token_trace)


# ============================================================================
# 4. Intent vs. Outcome Tracking Engine
# ============================================================================


class IntentClassificationRecord(StrictToolOutputModel):
    """Structured record of the user's parsed intent prior to orchestration."""

    intent_id: str = Field(..., description="Unique ID for this intent evaluation.")
    session_id: str = Field(default="default", description="Session identifier.")
    intent_category: Literal[
        "COA_CATALOG_SEARCH",
        "COA_CATALOG_AND_LOT_SEARCH",
        "LOT_COMPARISON",
        "CERTIFICATE_TYPE_INQUIRY",
        "INVALID_CATALOG_NUMBER",
        "HIGH_STAKES_REGULATORY_ACTION",
        "OUT_OF_SCOPE_NON_THERMOFISHER",
    ] = Field(..., description="Primary classified user intent.")
    extracted_catalog_number: Optional[str] = Field(
        default=None, description="Catalog number extracted from query or session context."
    )
    extracted_lot_numbers: list[str] = Field(
        default_factory=list, description="Lot number(s) extracted from query."
    )
    requires_hitl_confirmation: bool = Field(
        default=False, description="True if query requests a high-stakes GxP/LIMS release or bulk regulatory export."
    )
    expected_tools: list[str] = Field(
        default_factory=list, description="Tools expected to be invoked to fulfill this intent."
    )


class IntentOutcomeRecord(StrictToolOutputModel):
    """Audit record comparing user intent against actual agent outcome."""

    intent_id: str = Field(..., description="Correlated intent ID.")
    session_id: str = Field(..., description="Session identifier.")
    trace_id: Optional[str] = Field(default=None, description="Correlated distributed trace ID.")
    intent_category: str = Field(..., description="Classified user intent.")
    expected_tools: list[str] = Field(default_factory=list, description="Expected tools for the intent.")
    actual_tools_invoked: list[str] = Field(default_factory=list, description="Actual tools executed during the turn.")
    sub_agents_dispatched: list[str] = Field(default_factory=list, description="Sub-agents invoked.")
    selected_model: str = Field(..., description="Model selected by the strategic router.")
    outcome_status: Literal[
        "FULFILLED",
        "VALIDATION_GUIDANCE_RETURNED",
        "GUARDRAIL_BLOCKED_SCOPE",
        "GUARDRAIL_BLOCKED_SAFETY",
        "AWAITING_HITL_CONFIRMATION",
        "RECOVERED_FROM_TOOL_ERROR",
        "FAILED",
    ] = Field(..., description="Final resolution state of the user request.")
    goal_achieved: bool = Field(..., description="Whether the user's valid goal or policy boundary was satisfied.")
    tool_alignment_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overlap ratio between expected tools and actual tools invoked."
    )
    self_eval_score: float = Field(..., ge=0.0, le=1.0, description="Quality score from the self-evaluation critic.")
    drift_detected: bool = Field(
        default=False, description="True if the agent deviated from expected tools or failed self-eval."
    )
    latency_ms: float = Field(..., ge=0.0, description="Total end-to-end turn latency in milliseconds.")
    timestamp_utc: str = Field(..., description="UTC completion timestamp.")


class IntentOutcomeTracker:
    """Tracks user intent before execution and verifies outcome alignment after execution."""

    def __init__(self, logger: StructuredJSONLogger | None = None) -> None:
        self.logger = logger
        self.records: list[IntentOutcomeRecord] = []

    def classify_intent(
        self,
        prompt: str,
        *,
        session_id: str = "default",
        extracted_catalog: str | None = None,
        extracted_lots: list[str] | None = None,
        is_in_scope: bool = True,
    ) -> IntentClassificationRecord:
        """Deterministically classifies the user's intent and expected tool sequence."""
        lowered = prompt.lower()
        lots = extracted_lots or []

        if not is_in_scope:
            category: Any = "OUT_OF_SCOPE_NON_THERMOFISHER"
            expected = ["check_thermofisher_coa_scope"]
            requires_hitl = False
        elif any(
            kw in lowered
            for kw in (
                "lims release",
                "approve batch",
                "sign off",
                "bulk export",
                "smartsheet request",
                "submit missing coa",
                "commit release",
            )
        ):
            category = "HIGH_STAKES_REGULATORY_ACTION"
            expected = ["request_regulatory_action_confirmation"]
            requires_hitl = True
        elif any(kw in lowered for kw in ("certificate types", "types of certificates", "what certificate types")) and not extracted_catalog:
            category = "CERTIFICATE_TYPE_INQUIRY"
            expected = ["list_supported_certificate_types"]
            requires_hitl = False
        elif not extracted_catalog or len(extracted_catalog.strip()) < 2:
            category = "INVALID_CATALOG_NUMBER"
            expected = ["validate_catalog_and_lot_parameters"]
            requires_hitl = False
        elif len(lots) >= 2 or ("compare" in lowered and len(lots) >= 1):
            category = "LOT_COMPARISON"
            expected = [
                "validate_catalog_and_lot_parameters",
                "search_certificates_of_analysis",
                "compare_certificate_lots",
            ]
            requires_hitl = False
        elif len(lots) == 1:
            category = "COA_CATALOG_AND_LOT_SEARCH"
            expected = [
                "validate_catalog_and_lot_parameters",
                "search_certificates_of_analysis",
                "get_certificate_lot_details",
            ]
            requires_hitl = False
        else:
            category = "COA_CATALOG_SEARCH"
            expected = [
                "validate_catalog_and_lot_parameters",
                "search_certificates_of_analysis",
            ]
            requires_hitl = False

        return IntentClassificationRecord(
            intent_id=secrets.token_hex(6),
            session_id=session_id,
            intent_category=category,
            extracted_catalog_number=extracted_catalog,
            extracted_lot_numbers=lots,
            requires_hitl_confirmation=requires_hitl,
            expected_tools=expected,
        )

    def record_outcome(
        self,
        intent: IntentClassificationRecord,
        *,
        actual_tools_invoked: list[str],
        sub_agents_dispatched: list[str],
        selected_model: str,
        outcome_status: Literal[
            "FULFILLED",
            "VALIDATION_GUIDANCE_RETURNED",
            "GUARDRAIL_BLOCKED_SCOPE",
            "GUARDRAIL_BLOCKED_SAFETY",
            "AWAITING_HITL_CONFIRMATION",
            "RECOVERED_FROM_TOOL_ERROR",
            "FAILED",
        ],
        self_eval_score: float,
        latency_ms: float,
    ) -> IntentOutcomeRecord:
        """Computes alignment between classified intent and actual execution outcome."""
        expected_set = set(intent.expected_tools)
        actual_set = set(actual_tools_invoked)
        if not expected_set:
            alignment = 1.0
        else:
            alignment = round(len(expected_set & actual_set) / len(expected_set), 3)

        goal_achieved = outcome_status in {
            "FULFILLED",
            "VALIDATION_GUIDANCE_RETURNED",
            "GUARDRAIL_BLOCKED_SCOPE",
            "GUARDRAIL_BLOCKED_SAFETY",
            "AWAITING_HITL_CONFIRMATION",
        }
        drift_detected = (alignment < 0.5 and outcome_status == "FULFILLED") or self_eval_score < 0.75

        record = IntentOutcomeRecord(
            intent_id=intent.intent_id,
            session_id=intent.session_id,
            trace_id=_CURRENT_TRACE_ID.get(),
            intent_category=intent.intent_category,
            expected_tools=intent.expected_tools,
            actual_tools_invoked=actual_tools_invoked,
            sub_agents_dispatched=sub_agents_dispatched,
            selected_model=selected_model,
            outcome_status=outcome_status,
            goal_achieved=goal_achieved,
            tool_alignment_score=alignment,
            self_eval_score=round(self_eval_score, 3),
            drift_detected=drift_detected,
            latency_ms=round(latency_ms, 3),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        self.records.append(record)
        if self.logger is not None:
            self.logger.info(
                f"Intent vs Outcome recorded: {intent.intent_category} -> {outcome_status} (alignment={alignment})",
                component="intent_outcome_tracker",
                event_type="intent_outcome_audit",
                session_id=intent.session_id,
                latency_ms=latency_ms,
                attributes=record.model_dump(),
            )
        return record

    def get_metrics_summary(self) -> dict[str, Any]:
        """Aggregates intent-vs-outcome telemetry across all recorded turns."""
        total = len(self.records)
        if total == 0:
            return {
                "total_turns_tracked": 0,
                "goal_achievement_rate": 1.0,
                "average_tool_alignment_score": 1.0,
                "average_self_eval_score": 1.0,
                "drift_incidents": 0,
                "average_latency_ms": 0.0,
            }
        achieved = sum(1 for r in self.records if r.goal_achieved)
        avg_alignment = sum(r.tool_alignment_score for r in self.records) / total
        avg_eval = sum(r.self_eval_score for r in self.records) / total
        drift_count = sum(1 for r in self.records if r.drift_detected)
        avg_latency = sum(r.latency_ms for r in self.records) / total
        return {
            "total_turns_tracked": total,
            "goal_achievement_rate": round(achieved / total, 3),
            "average_tool_alignment_score": round(avg_alignment, 3),
            "average_self_eval_score": round(avg_eval, 3),
            "drift_incidents": drift_count,
            "average_latency_ms": round(avg_latency, 3),
        }


# ============================================================================
# Global Singletons & Instrumentation Decorators
# ============================================================================

pii_redactor = PIIRedactor()
structured_logger = StructuredJSONLogger(redactor=pii_redactor)
tracer = DistributedTracer(redactor=pii_redactor, logger=structured_logger)
intent_tracker = IntentOutcomeTracker(logger=structured_logger)


def trace_span(span_name: str, *, component: str = "tools") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that wraps sync or async functions in a distributed trace span with PII-safe argument capture."""

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer.start_as_current_span(
                    span_name,
                    component=component,
                    attributes={"function": fn.__name__, "kwargs": kwargs},
                ) as span:
                    result = await fn(*args, **kwargs)
                    if isinstance(result, StrictToolOutputModel) or isinstance(result, dict):
                        status_val = result.get("status")
                        if status_val == "error":
                            span.set_error(f"Tool returned guided error: {result.get('error_recovery')}")
                    return result

            return _async_wrapper

        @functools.wraps(fn)
        def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(
                span_name,
                component=component,
                attributes={"function": fn.__name__, "kwargs": kwargs},
            ) as span:
                result = fn(*args, **kwargs)
                if isinstance(result, StrictToolOutputModel) or isinstance(result, dict):
                    status_val = result.get("status")
                    if status_val == "error":
                        span.set_error(f"Tool returned guided error: {result.get('error_recovery')}")
                return result

        return _sync_wrapper

    return _decorator


__all__ = [
    "ActiveSpanHandle",
    "DistributedTracer",
    "HAS_OPENTELEMETRY",
    "IntentClassificationRecord",
    "IntentOutcomeRecord",
    "IntentOutcomeTracker",
    "PIIRedactionResult",
    "PIIRedactor",
    "StructuredJSONLogger",
    "StructuredLogEntry",
    "TelemetrySpanRecord",
    "intent_tracker",
    "pii_redactor",
    "structured_logger",
    "trace_span",
    "tracer",
]
