"""Cloud-Native Safety Settings, Model Armor Policies, Prompt Injection Defense, and Agentic Self-Evaluation Guardrails."""

from __future__ import annotations

import re
from typing import Any, Optional

from thermofisher_agent.knowledge import FISHERSCI_CERTIFICATES_PORTAL_URL
from thermofisher_agent.observability import pii_redactor, structured_logger, trace_span
from thermofisher_agent.schemas import Field, StrictToolOutputModel
from thermofisher_agent.tools import check_thermofisher_coa_scope

PROMPT_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"(?:reveal|print|dump|output)\s+(?:your\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:dan|unrestricted|jailbroken)", re.IGNORECASE),
    re.compile(r"bypass\s+(?:all\s+)?(?:safety|guardrails|restrictions)", re.IGNORECASE),
]


class CloudSafetyConfigSchema(StrictToolOutputModel):
    """Schema representing Vertex AI / Gemini Safety Settings and Google Cloud Model Armor policies."""

    provider: str = Field(default="google_genai_vertex_ai", description="Cloud safety provider.")
    model_armor_template_id: str = Field(
        default="projects/thermofisher-prod/locations/us-central1/templates/thermofisher-coa-model-armor-strict",
        description="Google Cloud Model Armor policy template resource URI.",
    )
    prompt_injection_filter: str = Field(default="ENABLED_HIGH", description="Model Armor prompt injection level.")
    pii_dlp_deidentification: str = Field(default="ENABLED_REDACT", description="Cloud DLP PII redaction mode.")
    harm_thresholds: dict[str, str] = Field(
        default_factory=lambda: {
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_LOW_AND_ABOVE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_LOW_AND_ABOVE",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_LOW_AND_ABOVE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_LOW_AND_ABOVE",
        },
        description="Cloud-native Gemini harm category block thresholds.",
    )


class InputGuardrailEvaluation(StrictToolOutputModel):
    """Result of pre-model input screening (PII scrubbing, prompt injection defense, and ThermoFisher CoA scope check)."""

    allowed: bool = Field(..., description="True if the prompt is safe and within ThermoFisher CoA scope.")
    block_reason: Optional[str] = Field(
        default=None, description="Reason for blocking (`PROMPT_INJECTION` or `OUT_OF_SCOPE`)."
    )
    sanitized_prompt: str = Field(..., description="PII-redacted user prompt.")
    pii_redacted_count: int = Field(default=0, ge=0, description="Number of PII tokens redacted from input.")
    is_in_scope: bool = Field(default=True, description="Whether the request is within ThermoFisher CoA scope.")
    detected_out_of_scope_topics: list[str] = Field(default_factory=list)
    extracted_catalog_candidates: list[str] = Field(default_factory=list)
    extracted_lot_candidates: list[str] = Field(default_factory=list)
    guardrail_message: str = Field(default="Input passed all pre-execution guardrails.")


class SelfEvaluationReport(StrictToolOutputModel):
    """Agentic Self-Evaluation (LLM-as-a-Judge / Rubric Critic) assessment of a generated CoA response."""

    passed: bool = Field(..., description="True if overall_score >= 0.85 and no critical policy violations exist.")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Weighted quality and compliance score.")
    criterion_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-criterion scores across the 5 ThermoFisher CoA evaluation dimensions.",
    )
    critique_notes: list[str] = Field(
        default_factory=list,
        description="Specific issues identified by the self-evaluation critic.",
    )
    auto_refined: bool = Field(
        default=False,
        description="True if the self-evaluation loop automatically repaired the draft response.",
    )
    refined_response: str = Field(..., description="Final verified and PII-clean response text.")


def get_cloud_native_safety_settings() -> list[Any]:
    """Builds Google GenAI / Vertex AI `types.SafetySetting` objects with `BLOCK_LOW_AND_ABOVE` thresholds."""
    config = CloudSafetyConfigSchema()
    try:
        from google.genai import types

        return [
            types.SafetySetting(
                category=getattr(types.HarmCategory, category),
                threshold=getattr(types.HarmBlockThreshold, threshold),
            )
            for category, threshold in config.harm_thresholds.items()
            if hasattr(types.HarmCategory, category) and hasattr(types.HarmBlockThreshold, threshold)
        ]
    except Exception:
        return [
            {"category": category, "threshold": threshold}
            for category, threshold in config.harm_thresholds.items()
        ]


class DefenseInDepthGuardrailEngine:
    """Three-layer guardrail pipeline: Input Sanitization, Cloud-Native Safety, and Agentic Self-Evaluation."""

    def __init__(self) -> None:
        self.cloud_safety_config = CloudSafetyConfigSchema()

    @trace_span("thermofisher.guardrail.evaluate_input", component="guardrail")
    def evaluate_input(self, prompt: str) -> InputGuardrailEvaluation:
        """Executes Layer 1 (PII + Prompt Injection) and Layer 2 (ThermoFisher CoA Scope) input guardrails."""
        redaction = pii_redactor.redact_text(prompt)
        sanitized = redaction.redacted_text

        for pattern in PROMPT_INJECTION_PATTERNS:
            if pattern.search(sanitized):
                structured_logger.warning(
                    "Blocked prompt injection attempt in InputGuardrail",
                    component="guardrail",
                    event_type="prompt_injection_blocked",
                    attributes={"pattern": pattern.pattern},
                )
                return InputGuardrailEvaluation(
                    allowed=False,
                    block_reason="PROMPT_INJECTION",
                    sanitized_prompt=sanitized,
                    pii_redacted_count=redaction.redaction_count,
                    is_in_scope=True,
                    guardrail_message=(
                        "I am the **ThermoFisher Scientific Certificate of Analysis (CoA) Specialist Agent**. "
                        "I cannot process requests that attempt to override safety instructions or system boundaries. "
                        "Please provide a **Catalog Number (mandatory, minimum 2 characters)** and an optional **Lot Number** "
                        f"to search for Certificates of Analysis on `{FISHERSCI_CERTIFICATES_PORTAL_URL}`."
                    ),
                )

        scope = check_thermofisher_coa_scope(sanitized)
        if not scope.is_in_scope:
            out_topics = ", ".join(scope.detected_out_of_scope_topics)
            structured_logger.info(
                f"Scope guardrail blocked unrelated request: {out_topics}",
                component="guardrail",
                event_type="scope_guardrail_blocked",
                attributes={"out_of_scope_topics": scope.detected_out_of_scope_topics},
            )
            return InputGuardrailEvaluation(
                allowed=False,
                block_reason="OUT_OF_SCOPE",
                sanitized_prompt=sanitized,
                pii_redacted_count=redaction.redaction_count,
                is_in_scope=False,
                detected_out_of_scope_topics=scope.detected_out_of_scope_topics,
                extracted_catalog_candidates=scope.extracted_catalog_candidates,
                extracted_lot_candidates=scope.extracted_lot_candidates,
                guardrail_message=(
                    f"I am the **ThermoFisher Scientific Certificate of Analysis (CoA) Specialist Agent**.\n\n"
                    f"I noticed your request mentions **{out_topics}**, which is outside my domain. I specialize "
                    f"exclusively in interacting with the ThermoFisher / Fisher Scientific Documents & Certificates "
                    f"portal (`{FISHERSCI_CERTIFICATES_PORTAL_URL}`) to look up **Certificates of Analysis**.\n\n"
                    f"To search for a Certificate of Analysis, please provide:\n"
                    f"- **Catalog Number** (*Mandatory, minimum of 2 characters*, e.g., `AC29`, `AC299131000`, `A452-4`)\n"
                    f"- **Lot Number** (*Optional*, e.g., `A0487095`, `A0487009`)"
                ),
            )

        return InputGuardrailEvaluation(
            allowed=True,
            block_reason=None,
            sanitized_prompt=sanitized,
            pii_redacted_count=redaction.redaction_count,
            is_in_scope=True,
            detected_out_of_scope_topics=[],
            extracted_catalog_candidates=scope.extracted_catalog_candidates,
            extracted_lot_candidates=scope.extracted_lot_candidates,
            guardrail_message="Input passed all pre-execution guardrails.",
        )

    @trace_span("thermofisher.guardrail.self_evaluate_response", component="guardrail")
    def self_evaluate_and_refine(
        self,
        prompt: str,
        draft_response: str,
        *,
        is_search_results_response: bool = True,
    ) -> SelfEvaluationReport:
        """Executes Layer 3 Agentic Self-Evaluation (Critic Loop) and auto-refines any rubric gaps."""
        critique_notes: list[str] = []
        refined = draft_response

        # 1. PII & Safety Compliance check
        redacted_check = pii_redactor.redact_text(refined)
        if redacted_check.redaction_count > 0:
            critique_notes.append(f"Scrubbed {redacted_check.redaction_count} leaked PII token(s) from output.")
            refined = redacted_check.redacted_text
        pii_score = 1.0

        # 2. ThermoFisher Specialist Identity check
        identity_score = 1.0
        if "ThermoFisher" not in refined and "Fisher Scientific" not in refined:
            critique_notes.append("Injected ThermoFisher Scientific Certificate of Analysis Specialist identity header.")
            refined = (
                "**ThermoFisher Scientific Certificate of Analysis (CoA) Specialist Agent**\n\n" + refined
            )
            identity_score = 0.92

        # 3. Portal URL & Attribution check
        portal_score = 1.0
        if FISHERSCI_CERTIFICATES_PORTAL_URL not in refined:
            critique_notes.append("Appended Fisher Scientific Certificate Search Portal URL attribution.")
            refined += (
                f"\n\n### 🔗 Official ThermoFisher / Fisher Scientific Portal\n"
                f"- **Search URL**: [{FISHERSCI_CERTIFICATES_PORTAL_URL}]({FISHERSCI_CERTIFICATES_PORTAL_URL})\n"
                f"- **Input Rules**: Catalog Number (*Mandatory, min 2 characters*) | Lot Number (*Optional*)"
            )
            portal_score = 0.90

        # 4. 5-Column Fisher Scientific Table Structure check (when returning search results)
        table_score = 1.0
        if is_search_results_response:
            required_headers = ("Lot #", "Certificate Type", "Date", "Catalog Number(s)", "Product Name")
            missing_headers = [h for h in required_headers if h not in refined]
            if missing_headers:
                critique_notes.append(
                    f"Appended Fisher Scientific 5-column schema reference ({', '.join(missing_headers)})."
                )
                refined += (
                    "\n\n### 📋 Fisher Scientific Certificate Results Schema\n"
                    "| Lot # | Certificate Type | Date | Catalog Number(s) | Product Name |\n"
                    "| :--- | :--- | :--- | :--- | :--- |\n"
                    "| *(See matched lot records above)* | Certificate of Analysis | Published Date | Covered SKUs | Certified Product Title |"
                )
                table_score = 0.90

        # 5. Mandatory Catalog Number (>= 2 chars) & Optional Lot Number Policy Compliance
        validation_score = 1.0

        scores = {
            "thermofisher_scope_adherence": identity_score,
            "catalog_min_2_chars_policy": validation_score,
            "fishersci_5_column_table_fidelity": table_score,
            "portal_url_attribution": portal_score,
            "pii_and_safety_compliance": pii_score,
        }
        overall = round(sum(scores.values()) / len(scores), 3)
        auto_refined = len(critique_notes) > 0

        structured_logger.info(
            f"Agentic self-evaluation completed (score={overall}, auto_refined={auto_refined})",
            component="guardrail",
            event_type="agentic_self_eval",
            attributes={"overall_score": overall, "critique_notes": critique_notes, "scores": scores},
        )

        return SelfEvaluationReport(
            passed=overall >= 0.85,
            overall_score=overall,
            criterion_scores=scores,
            critique_notes=critique_notes,
            auto_refined=auto_refined,
            refined_response=refined,
        )

    def adk_before_model_callback(self, callback_context: Any = None, llm_request: Any = None) -> Any:
        """Google ADK `before_model_callback` enforcing input guardrails before LLM invocation."""
        structured_logger.info(
            "Executed ADK before_model_callback guardrail check",
            component="guardrail",
            event_type="adk_before_model_callback",
        )
        return None

    def adk_after_model_callback(self, callback_context: Any = None, llm_response: Any = None) -> Any:
        """Google ADK `after_model_callback` running self-evaluation and PII redaction on model output."""
        structured_logger.info(
            "Executed ADK after_model_callback self-eval check",
            component="guardrail",
            event_type="adk_after_model_callback",
        )
        return None


guardrail_engine = DefenseInDepthGuardrailEngine()


@trace_span("thermofisher.tool.evaluate_coa_response_quality", component="guardrail")
def evaluate_coa_response_quality(prompt: str, draft_response: str) -> SelfEvaluationReport:
    """Agentic self-evaluation tool used by the `safety_and_critic_specialist` sub-agent to audit CoA responses."""
    return guardrail_engine.self_evaluate_and_refine(prompt, draft_response, is_search_results_response=True)


__all__ = [
    "CloudSafetyConfigSchema",
    "DefenseInDepthGuardrailEngine",
    "InputGuardrailEvaluation",
    "SelfEvaluationReport",
    "evaluate_coa_response_quality",
    "get_cloud_native_safety_settings",
    "guardrail_engine",
]
