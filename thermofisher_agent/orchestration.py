"""Multi-Agent Hierarchy (Coordinator + Specialist Sub-Agents) and Strategic Model Routing for ThermoFisher CoA Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from thermofisher_agent.guardrails import evaluate_coa_response_quality
from thermofisher_agent.hitl import hitl_manager
from thermofisher_agent.observability import structured_logger, trace_span
from thermofisher_agent.schemas import Field, StrictToolOutputModel
from thermofisher_agent.tools import (
    check_thermofisher_coa_scope,
    compare_certificate_lots,
    get_certificate_lot_details,
    list_supported_certificate_types,
    request_regulatory_action_confirmation,
    resolve_regulatory_action_confirmation,
    search_certificates_of_analysis,
    validate_catalog_and_lot_parameters,
)

try:
    from google.adk.agents import Agent as AdkAgent  # type: ignore[import-untyped]

    HAS_GOOGLE_ADK = True
except ImportError:
    HAS_GOOGLE_ADK = False

    @dataclass
    class AdkAgent:  # type: ignore[no-redef]
        """Fallback Agent representation matching google.adk.agents.Agent interface with sub_agents and callbacks."""

        name: str
        model: str
        description: str
        instruction: str
        tools: list[Callable[..., Any]] = field(default_factory=list)
        sub_agents: list[Any] = field(default_factory=list)
        before_model_callback: Callable[..., Any] | None = None
        after_model_callback: Callable[..., Any] | None = None
        before_tool_callback: Callable[..., Any] | None = None


# ============================================================================
# 1. Strategic Model Router (Flash-Lite vs. Flash vs. Pro)
# ============================================================================

MODEL_TIER_FAST = "gemini-2.5-flash-lite"
MODEL_TIER_STANDARD = "gemini-2.5-flash"
MODEL_TIER_REASONING = "gemini-2.5-pro"


class ModelRoutingDecision(StrictToolOutputModel):
    """Structured output of the Strategic Model Router."""

    selected_model: str = Field(..., description="Selected Gemini model identifier.")
    tier: Literal["fast", "standard", "reasoning"] = Field(
        ..., description="Routing tier ('fast', 'standard', or 'reasoning')."
    )
    complexity_score: float = Field(..., ge=0.0, le=1.0, description="Normalized query complexity score.")
    routing_reason: str = Field(..., description="Explanation of why this model tier was chosen.")
    fallback_models: list[str] = Field(
        default_factory=list,
        description="Ordered fallback models if the primary model experiences quota or latency issues.",
    )


class StrategicModelRouter:
    """Routes queries dynamically across Gemini model tiers based on task complexity and cost/latency targets."""

    def __init__(
        self,
        fast_model: str = MODEL_TIER_FAST,
        standard_model: str = MODEL_TIER_STANDARD,
        reasoning_model: str = MODEL_TIER_REASONING,
    ) -> None:
        self.fast_model = fast_model
        self.standard_model = standard_model
        self.reasoning_model = reasoning_model

    @trace_span("thermofisher.router.route_query", component="router")
    def route_query(
        self,
        prompt: str,
        *,
        task_type: str | None = None,
        lot_count: int = 0,
        has_valid_catalog: bool = True,
    ) -> ModelRoutingDecision:
        """Selects the optimal Gemini model tier for a given prompt and task category."""
        lowered = prompt.lower()

        if task_type in {"scope_check", "compaction", "certificate_types", "validation_only"}:
            return ModelRoutingDecision(
                selected_model=self.fast_model,
                tier="fast",
                complexity_score=0.2,
                routing_reason=f"Task '{task_type}' is a bounded low-latency lookup suited for {self.fast_model}.",
                fallback_models=[self.standard_model],
            )

        score = 0.25
        if lot_count >= 2:
            score += 0.35
        elif lot_count == 1:
            score += 0.15

        if has_valid_catalog:
            score += 0.15

        complex_keywords = (
            "compare",
            "audit",
            "regulatory",
            "gmp",
            "gxp",
            "iso",
            "lims",
            "difference",
            "multiple lots",
            "batch release",
            "analyze",
        )
        matched_kw = sum(1 for kw in complex_keywords if kw in lowered)
        score += min(0.35, matched_kw * 0.10)
        score = round(min(1.0, score), 2)

        if score >= 0.65 or lot_count >= 2 or ("compare" in lowered and has_valid_catalog):
            decision = ModelRoutingDecision(
                selected_model=self.reasoning_model,
                tier="reasoning",
                complexity_score=score,
                routing_reason=(
                    f"High-complexity multi-lot comparison or regulatory audit synthesis "
                    f"(lots={lot_count}, score={score}) routed to deep-reasoning model {self.reasoning_model}."
                ),
                fallback_models=[self.standard_model, self.fast_model],
            )
        elif score >= 0.35 or has_valid_catalog:
            decision = ModelRoutingDecision(
                selected_model=self.standard_model,
                tier="standard",
                complexity_score=score,
                routing_reason=(
                    f"Standard Certificate of Analysis search & verification (score={score}) "
                    f"routed to balanced model {self.standard_model}."
                ),
                fallback_models=[self.fast_model, self.reasoning_model],
            )
        else:
            decision = ModelRoutingDecision(
                selected_model=self.fast_model,
                tier="fast",
                complexity_score=score,
                routing_reason=f"Quick validation or certificate type lookup (score={score}) routed to {self.fast_model}.",
                fallback_models=[self.standard_model],
            )

        structured_logger.info(
            f"Strategic router selected {decision.selected_model} (tier={decision.tier}, score={decision.complexity_score})",
            component="router",
            event_type="model_routed",
            attributes=decision.model_dump(),
        )
        return decision


model_router = StrategicModelRouter()


# ============================================================================
# 2. Multi-Agent Specialist Sub-Agents Definition (Google ADK Hierarchy)
# ============================================================================


def _instantiate_adk_agent(
    *,
    name: str,
    model: str,
    description: str,
    instruction: str,
    tools: list[Callable[..., Any]],
    sub_agents: list[Any] | None = None,
    before_model_callback: Callable[..., Any] | None = None,
    after_model_callback: Callable[..., Any] | None = None,
    before_tool_callback: Callable[..., Any] | None = None,
) -> AdkAgent:
    """Safely constructs a Google ADK Agent with sub_agents and lifecycle callbacks."""
    kwargs: dict[str, Any] = {
        "name": name,
        "model": model,
        "description": description,
        "instruction": instruction,
        "tools": tools,
    }
    if sub_agents:
        kwargs["sub_agents"] = sub_agents
    if before_model_callback is not None:
        kwargs["before_model_callback"] = before_model_callback
    if after_model_callback is not None:
        kwargs["after_model_callback"] = after_model_callback
    if before_tool_callback is not None:
        kwargs["before_tool_callback"] = before_tool_callback

    try:
        return AdkAgent(**kwargs)
    except TypeError:
        agent = AdkAgent(
            name=name,
            model=model,
            description=description,
            instruction=instruction,
            tools=tools,
        )
        if sub_agents is not None:
            setattr(agent, "sub_agents", sub_agents)
        if before_model_callback is not None:
            setattr(agent, "before_model_callback", before_model_callback)
        if after_model_callback is not None:
            setattr(agent, "after_model_callback", after_model_callback)
        if before_tool_callback is not None:
            setattr(agent, "before_tool_callback", before_tool_callback)
        return agent


def create_specialist_sub_agents() -> dict[str, AdkAgent]:
    """Creates the 4 specialized sub-agents for the ThermoFisher CoA multi-agent architecture."""
    coa_search_agent = _instantiate_adk_agent(
        name="coa_search_specialist",
        model=MODEL_TIER_STANDARD,
        description=(
            "Specialist sub-agent for querying ThermoFisher / Fisher Scientific "
            "(https://www.fishersci.com/search/documents?type=certificates) by mandatory Catalog Number "
            "(minimum 2 characters) and optional Lot Number."
        ),
        instruction=(
            "You are the ThermoFisher Certificate of Analysis Search Specialist Sub-Agent. Always invoke "
            "`validate_catalog_and_lot_parameters` and `search_certificates_of_analysis`. If a tool returns "
            "`status='error'`, follow `error_recovery.llm_recovery_instruction` to self-correct or ask the user "
            "for a Catalog Number of at least 2 characters."
        ),
        tools=[validate_catalog_and_lot_parameters, search_certificates_of_analysis],
    )

    lot_verification_agent = _instantiate_adk_agent(
        name="lot_verification_and_comparison_specialist",
        model=MODEL_TIER_REASONING,
        description=(
            "Specialist sub-agent for inspecting individual manufacturing Lot # records, expanding multi-SKU "
            "Catalog Number(s) ('View All'), and comparing multiple lots side by side."
        ),
        instruction=(
            "You are the Lot Verification & Comparison Specialist Sub-Agent. Use `get_certificate_lot_details` "
            "to inspect specific lot records and `compare_certificate_lots` to compare two or more lots."
        ),
        tools=[get_certificate_lot_details, compare_certificate_lots],
    )

    regulatory_and_hitl_agent = _instantiate_adk_agent(
        name="regulatory_and_hitl_specialist",
        model=MODEL_TIER_STANDARD,
        description=(
            "Specialist sub-agent for Fisher Scientific certificate types (CoA, Sterility, Compliance, Animal Origin, "
            "Validation, Guarantee, REACH) and HITL-gated GxP/LIMS release or bulk export workflows."
        ),
        instruction=(
            "You are the Regulatory & HITL Specialist Sub-Agent. Use `list_supported_certificate_types` to explain "
            "certificate categories. For any LIMS batch release, bulk export, or external missing-CoA request, "
            "ALWAYS invoke `request_regulatory_action_confirmation` and wait for human approval."
        ),
        tools=[
            list_supported_certificate_types,
            request_regulatory_action_confirmation,
            resolve_regulatory_action_confirmation,
        ],
        before_tool_callback=hitl_manager.adk_before_tool_callback,
    )

    safety_and_critic_agent = _instantiate_adk_agent(
        name="safety_and_critic_specialist",
        model=MODEL_TIER_FAST,
        description=(
            "Guardrail and self-evaluation critic sub-agent that enforces ThermoFisher CoA scope, mandatory "
            ">=2 char Catalog Number rules, and verifies the 5-column Fisher Scientific table structure."
        ),
        instruction=(
            "You are the Safety & Self-Evaluation Critic Sub-Agent. Use `check_thermofisher_coa_scope` to block "
            "out-of-scope requests and `evaluate_coa_response_quality` to audit draft responses."
        ),
        tools=[check_thermofisher_coa_scope, evaluate_coa_response_quality],
    )

    return {
        "coa_search_specialist": coa_search_agent,
        "lot_verification_and_comparison_specialist": lot_verification_agent,
        "regulatory_and_hitl_specialist": regulatory_and_hitl_agent,
        "safety_and_critic_specialist": safety_and_critic_agent,
    }


__all__ = [
    "AdkAgent",
    "HAS_GOOGLE_ADK",
    "MODEL_TIER_FAST",
    "MODEL_TIER_REASONING",
    "MODEL_TIER_STANDARD",
    "ModelRoutingDecision",
    "StrategicModelRouter",
    "_instantiate_adk_agent",
    "create_specialist_sub_agents",
    "model_router",
]
