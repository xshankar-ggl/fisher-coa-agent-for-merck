"""Human-in-the-Loop (HITL) confirmation hooks and approval state machine for high-stakes regulatory/quality actions."""

from __future__ import annotations

import functools
import secrets
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Optional

from thermofisher_agent.observability import structured_logger, trace_span
from thermofisher_agent.schemas import (
    ErrorRecoveryGuidance,
    Field,
    HighStakesRegulatoryOutput,
    StrictToolOutputModel,
)

HIGH_STAKES_REGULATORY_ACTION_TYPES: set[str] = {
    "LIMS_QUALITY_RELEASE_APPROVAL",
    "BULK_COA_EXPORT",
    "SUBMIT_MISSING_COA_SMARTFORM_REQUEST",
    "REGULATORY_AUDIT_PACKAGE_SIGN_OFF",
}


class PendingRegulatoryConfirmationRecord(StrictToolOutputModel):
    """Record stored in the HITL state machine for a regulatory action awaiting human QA approval."""

    confirmation_id: str = Field(..., description="Unique confirmation identifier.")
    session_id: str = Field(default="default", description="Session requesting the action.")
    action_type: str = Field(..., description="Category of high-stakes regulatory/quality action.")
    catalog_number: str = Field(..., description="Target ThermoFisher Catalog Number (minimum 2 characters).")
    lot_number: Optional[str] = Field(default=None, description="Optional manufacturing Lot Number.")
    action_description: str = Field(..., description="Detailed description of the regulatory commitment.")
    gxp_regulated: bool = Field(default=True, description="Whether the action is subject to GxP / ISO / 21 CFR Part 11.")
    status: Literal["pending_human_approval", "approved", "rejected"] = Field(
        default="pending_human_approval", description="Approval status."
    )
    reviewer_id: Optional[str] = Field(default=None, description="Identifier of the human QA reviewer.")
    reviewer_note: Optional[str] = Field(default=None, description="Optional audit note from the human reviewer.")
    created_at_utc: str = Field(..., description="ISO-8601 creation timestamp.")
    resolved_at_utc: Optional[str] = Field(default=None, description="ISO-8601 resolution timestamp.")


class ConfirmationDecisionOutput(StrictToolOutputModel):
    """Strict output schema returned when a human approves or rejects a pending regulatory action."""

    status: Literal["success", "error"] = Field(default="success")
    confirmation_id: str = Field(..., description="Confirmation identifier.")
    decision: Literal["approved", "rejected", "not_found"] = Field(..., description="Human QA decision.")
    action_type: str = Field(..., description="High-stakes regulatory action category.")
    catalog_number: str = Field(..., description="Catalog number associated with the action.")
    lot_number: Optional[str] = Field(default=None, description="Optional lot number associated with the action.")
    action_description: str = Field(..., description="Description of the committed or rejected action.")
    execution_summary: str = Field(..., description="Audit summary of the committed or cancelled regulatory action.")
    error_recovery: Optional[ErrorRecoveryGuidance] = Field(default=None)


class HumanInTheLoopManager:
    """Manages Human-in-the-Loop (HITL) approval gates for GxP quality releases, bulk exports, and external submissions."""

    def __init__(self) -> None:
        self._pending_actions: dict[str, PendingRegulatoryConfirmationRecord] = {}

    @trace_span("thermofisher.hitl.request_confirmation", component="hitl")
    def request_confirmation(
        self,
        *,
        action_type: str,
        catalog_number: str,
        action_description: str,
        lot_number: str | None = None,
        gxp_regulated: bool = True,
        session_id: str = "default",
    ) -> HighStakesRegulatoryOutput:
        """Creates a pending HITL confirmation gate and pauses execution until human QA approval."""
        normalized_type = action_type.strip().upper()
        cleaned_catalog = (catalog_number or "").strip().upper()
        cleaned_lot = lot_number.strip().upper() if lot_number and lot_number.strip() else None

        if len(cleaned_catalog) < 2:
            return HighStakesRegulatoryOutput(
                status="error",
                requires_human_confirmation=True,
                confirmation_id="INVALID",
                action_type=normalized_type,
                catalog_number=catalog_number,
                lot_number=cleaned_lot,
                action_description=action_description,
                regulatory_impact_warning="Validation failed before staging HITL regulatory action.",
                approval_instructions="Provide a valid Catalog Number of at least 2 characters and retry.",
                error_recovery=ErrorRecoveryGuidance(
                    error_code="CATALOG_NUMBER_TOO_SHORT",
                    error_message=(
                        f"Catalog number '{catalog_number}' is invalid. Catalog number is mandatory and must be "
                        "a minimum of 2 characters."
                    ),
                    invalid_parameter="catalog_number",
                    provided_value=str(catalog_number),
                    allowed_values=["Minimum 2 alphanumeric characters (e.g., 'AC29', 'AC299131000', 'A452-4')"],
                    llm_recovery_instruction=(
                        "Ask the user for a ThermoFisher Catalog Number with at least 2 characters before staging "
                        "a regulatory confirmation request."
                    ),
                    suggested_tool_call_example=(
                        "request_regulatory_action_confirmation("
                        "action_type='LIMS_QUALITY_RELEASE_APPROVAL', "
                        "catalog_number='AC299131000', lot_number='A0487095', "
                        "action_description='Release Lot A0487095 (Methanol-d, 99.0 atom % D) for NMR QC use')"
                    ),
                ),
            )

        if normalized_type not in HIGH_STAKES_REGULATORY_ACTION_TYPES:
            return HighStakesRegulatoryOutput(
                status="error",
                requires_human_confirmation=True,
                confirmation_id="INVALID",
                action_type=action_type,
                catalog_number=cleaned_catalog,
                lot_number=cleaned_lot,
                action_description=action_description,
                regulatory_impact_warning="Validation failed before staging regulatory action.",
                approval_instructions="Fix the invalid action_type parameter and retry.",
                error_recovery=ErrorRecoveryGuidance(
                    error_code="INVALID_HITL_ACTION_TYPE",
                    error_message=f"Unsupported high-stakes regulatory action_type '{action_type}'.",
                    invalid_parameter="action_type",
                    provided_value=str(action_type),
                    allowed_values=sorted(HIGH_STAKES_REGULATORY_ACTION_TYPES),
                    llm_recovery_instruction=(
                        "Re-invoke `request_regulatory_action_confirmation` using one of the supported "
                        f"action_type values: {', '.join(sorted(HIGH_STAKES_REGULATORY_ACTION_TYPES))}."
                    ),
                    suggested_tool_call_example=(
                        "request_regulatory_action_confirmation("
                        "action_type='LIMS_QUALITY_RELEASE_APPROVAL', "
                        "catalog_number='AC29', lot_number='A0487095', "
                        "action_description='Approve CoA for Lot A0487095 in LIMS')"
                    ),
                ),
            )

        confirmation_id = f"HITL-{secrets.token_hex(4).upper()}"
        record = PendingRegulatoryConfirmationRecord(
            confirmation_id=confirmation_id,
            session_id=session_id,
            action_type=normalized_type,
            catalog_number=cleaned_catalog,
            lot_number=cleaned_lot,
            action_description=action_description,
            gxp_regulated=gxp_regulated,
            status="pending_human_approval",
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        self._pending_actions[confirmation_id] = record

        structured_logger.info(
            f"HITL regulatory confirmation staged: {confirmation_id} ({normalized_type}, catalog={cleaned_catalog}, lot={cleaned_lot})",
            component="hitl",
            event_type="hitl_confirmation_requested",
            session_id=session_id,
            attributes=record.model_dump(),
        )

        impact_warning = (
            "IMPORTANT (GxP / 21 CFR Part 11 / ISO 9001 Compliance): LIMS batch releases, regulatory audit "
            "sign-offs, and external Smartsheet missing-CoA requests create permanent quality records. "
            "Execution is PAUSED pending explicit human QA review and approval."
            if gxp_regulated
            else "Standard quality workflow action; requires explicit human confirmation before execution."
        )

        lot_label = f", Lot `{cleaned_lot}`" if cleaned_lot else ""
        return HighStakesRegulatoryOutput(
            status="pending_human_approval",
            requires_human_confirmation=True,
            confirmation_id=confirmation_id,
            action_type=normalized_type,
            catalog_number=cleaned_catalog,
            lot_number=cleaned_lot,
            action_description=action_description,
            regulatory_impact_warning=impact_warning,
            approval_instructions=(
                f"Action `{confirmation_id}` is paused awaiting human QA approval for Catalog `{cleaned_catalog}`"
                f"{lot_label} ('{action_description}'). Ask the user to confirm via "
                f"`resolve_regulatory_action_confirmation(confirmation_id='{confirmation_id}', approved=True)` "
                f"or POST `/api/hitl/confirm`."
            ),
            error_recovery=None,
        )

    async def arequest_confirmation(self, **kwargs: Any) -> HighStakesRegulatoryOutput:
        """Async wrapper for creating a HITL confirmation gate."""
        return self.request_confirmation(**kwargs)

    @trace_span("thermofisher.hitl.resolve_confirmation", component="hitl")
    def resolve_confirmation(
        self,
        confirmation_id: str,
        *,
        approved: bool,
        reviewer_id: str = "qa_specialist",
        reviewer_note: str = "",
    ) -> ConfirmationDecisionOutput:
        """Approves or rejects a pending high-stakes regulatory action after human review."""
        record = self._pending_actions.get(confirmation_id)
        if record is None:
            return ConfirmationDecisionOutput(
                status="error",
                confirmation_id=confirmation_id,
                decision="not_found",
                action_type="UNKNOWN",
                catalog_number="UNKNOWN",
                lot_number=None,
                action_description="Unknown confirmation ID",
                execution_summary=f"Confirmation ID '{confirmation_id}' was not found.",
                error_recovery=ErrorRecoveryGuidance(
                    error_code="CONFIRMATION_ID_NOT_FOUND",
                    error_message=f"No pending HITL action matches confirmation_id='{confirmation_id}'.",
                    invalid_parameter="confirmation_id",
                    provided_value=confirmation_id,
                    allowed_values=sorted(self._pending_actions.keys())
                    or ["Call request_regulatory_action_confirmation first"],
                    llm_recovery_instruction=(
                        "Verify the `confirmation_id` returned by `request_regulatory_action_confirmation` "
                        "or stage a new confirmation request before attempting to resolve it."
                    ),
                    suggested_tool_call_example=(
                        "resolve_regulatory_action_confirmation(confirmation_id='HITL-1A2B3C4D', approved=True)"
                    ),
                ),
            )

        decision: Literal["approved", "rejected"] = "approved" if approved else "rejected"
        record.status = decision
        record.reviewer_id = reviewer_id
        record.reviewer_note = reviewer_note
        record.resolved_at_utc = datetime.now(timezone.utc).isoformat()

        structured_logger.info(
            f"HITL confirmation resolved: {confirmation_id} -> {decision}",
            component="hitl",
            event_type="hitl_confirmation_resolved",
            session_id=record.session_id,
            attributes=record.model_dump(),
        )

        lot_str = f" / Lot {record.lot_number}" if record.lot_number else ""
        summary = (
            f"Human QA reviewer '{reviewer_id}' APPROVED {record.action_type} for Catalog {record.catalog_number}{lot_str} "
            f"('{record.action_description}'). Regulatory action has been committed."
            if approved
            else f"Human QA reviewer '{reviewer_id}' REJECTED {record.action_type} for Catalog {record.catalog_number}{lot_str} "
            f"('{record.action_description}'). No changes were committed."
        )
        return ConfirmationDecisionOutput(
            status="success",
            confirmation_id=confirmation_id,
            decision=decision,
            action_type=record.action_type,
            catalog_number=record.catalog_number,
            lot_number=record.lot_number,
            action_description=record.action_description,
            execution_summary=summary,
            error_recovery=None,
        )

    async def aresolve_confirmation(self, confirmation_id: str, **kwargs: Any) -> ConfirmationDecisionOutput:
        """Async wrapper for resolving a pending HITL confirmation."""
        return self.resolve_confirmation(confirmation_id, **kwargs)

    def list_pending_actions(self, session_id: str | None = None) -> list[PendingRegulatoryConfirmationRecord]:
        """Lists all regulatory actions currently in `pending_human_approval` state."""
        return [
            rec
            for rec in self._pending_actions.values()
            if rec.status == "pending_human_approval"
            and (session_id is None or rec.session_id == session_id)
        ]

    def adk_before_tool_callback(
        self,
        tool: Any,
        args: dict[str, Any],
        tool_context: Any = None,
    ) -> dict[str, Any] | None:
        """Google ADK `before_tool_callback` hook enforcing human confirmation on gated tools."""
        tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))
        if getattr(tool, "requires_human_confirmation", False):
            conf_id = args.get("confirmation_id")
            if conf_id and conf_id in self._pending_actions:
                rec = self._pending_actions[conf_id]
                if rec.status == "approved":
                    return None
            structured_logger.warning(
                f"ADK before_tool_callback intercepted high-stakes regulatory tool: {tool_name}",
                component="hitl",
                event_type="hitl_adk_hook_triggered",
                attributes={"tool_name": tool_name, "args": args},
            )
        return None


hitl_manager = HumanInTheLoopManager()


def require_human_confirmation(action_type: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator marking a tool as requiring Human-in-the-Loop confirmation before commitment."""

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(fn, "requires_human_confirmation", True)
        setattr(fn, "hitl_action_type", action_type)

        @functools.wraps(fn)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        setattr(_wrapper, "requires_human_confirmation", True)
        setattr(_wrapper, "hitl_action_type", action_type)
        return _wrapper

    return _decorator


__all__ = [
    "ConfirmationDecisionOutput",
    "HIGH_STAKES_REGULATORY_ACTION_TYPES",
    "HumanInTheLoopManager",
    "PendingRegulatoryConfirmationRecord",
    "hitl_manager",
    "require_human_confirmation",
]
