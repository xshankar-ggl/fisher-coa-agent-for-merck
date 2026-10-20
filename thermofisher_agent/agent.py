"""Google ADK Multi-Agent Coordinator, Async Execution Engine, Strategic Router, and Guardrail Pipeline."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from thermofisher_agent.guardrails import (
    get_cloud_native_safety_settings,
    guardrail_engine,
)
from thermofisher_agent.hitl import hitl_manager
from thermofisher_agent.knowledge import FISHERSCI_CERTIFICATES_PORTAL_URL
from thermofisher_agent.memory import (
    PersistentSessionMemoryManager,
    session_memory_manager,
)
from thermofisher_agent.observability import (
    intent_tracker,
    structured_logger,
    trace_span,
    tracer,
)
from thermofisher_agent.orchestration import (
    AdkAgent,
    MODEL_TIER_STANDARD,
    _instantiate_adk_agent,
    create_specialist_sub_agents,
    model_router,
)
from thermofisher_agent.prompt import (
    THERMOFISHER_ROLE_HEADER,
    THERMOFISHER_SYSTEM_PROMPT,
)
from thermofisher_agent.secrets import secret_manager_service
from thermofisher_agent.tools import (
    ALL_THERMOFISHER_TOOLS,
    compare_certificate_lots,
    extract_catalog_and_lot_from_text,
    get_certificate_lot_details,
    list_supported_certificate_types,
    request_regulatory_action_confirmation,
    search_certificates_of_analysis,
    validate_catalog_and_lot_parameters,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = os.getenv("THERMOFISHER_MODEL", MODEL_TIER_STANDARD)


def create_thermofisher_agent(model: str = DEFAULT_MODEL) -> AdkAgent:
    """Creates and returns the hierarchical ThermoFisher CoA Google ADK `root_agent` Coordinator with 4 specialist sub-agents."""
    sub_agents_map = create_specialist_sub_agents()
    return _instantiate_adk_agent(
        name="thermofisher_coa_agent",
        model=model,
        description=(
            "ThermoFisher Scientific Certificate of Analysis (CoA) Specialist Agent: Coordinates 4 sub-agents "
            "to search, verify, and compare Certificates of Analysis on https://www.fishersci.com/search/documents?type=certificates "
            "by mandatory Catalog Number (>= 2 chars) and optional Lot Number."
        ),
        instruction=THERMOFISHER_SYSTEM_PROMPT,
        tools=ALL_THERMOFISHER_TOOLS,
        sub_agents=list(sub_agents_map.values()),
        before_model_callback=guardrail_engine.adk_before_model_callback,
        after_model_callback=guardrail_engine.adk_after_model_callback,
        before_tool_callback=hitl_manager.adk_before_tool_callback,
    )


# Standard Google ADK entrypoint symbol (`adk run thermofisher_agent` / `adk web`)
root_agent = create_thermofisher_agent()


@trace_span("thermofisher.agent.render_tool_assisted_response", component="agent")
def render_tool_assisted_response(
    prompt: str,
    *,
    session_active_catalog: str | None = None,
    memory_context_summary: str = "",
    prefer_live_api: bool = True,
) -> str:
    """Synthesizes a comprehensive ThermoFisher CoA response by coordinating the specialist tools."""
    input_eval = guardrail_engine.evaluate_input(prompt)
    if not input_eval.allowed:
        return input_eval.guardrail_message

    sanitized_prompt = input_eval.sanitized_prompt
    lowered = sanitized_prompt.lower()

    extracted_cat, extracted_lots = extract_catalog_and_lot_from_text(sanitized_prompt)
    resolved_catalog = extracted_cat or session_active_catalog

    # 1. Check if the user is requesting a high-stakes GxP/LIMS regulatory action (HITL Gate)
    if any(
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
        action_type = (
            "BULK_COA_EXPORT"
            if "bulk export" in lowered
            else "SUBMIT_MISSING_COA_SMARTFORM_REQUEST"
            if "smartsheet" in lowered or "missing coa" in lowered
            else "REGULATORY_AUDIT_PACKAGE_SIGN_OFF"
            if "audit" in lowered or "sign off" in lowered
            else "LIMS_QUALITY_RELEASE_APPROVAL"
        )
        cat_for_hitl = resolved_catalog or "AC29"
        lot_for_hitl = extracted_lots[0] if extracted_lots else None
        hitl_res = request_regulatory_action_confirmation(
            action_type=action_type,
            catalog_number=cat_for_hitl,
            lot_number=lot_for_hitl,
            action_description=f"User requested regulatory action: {sanitized_prompt[:120]}",
            gxp_regulated=True,
        )
        return (
            f"# 🛑 Human-in-the-Loop QA Confirmation Required (`{hitl_res.confirmation_id}`)\n\n"
            f"I am the **ThermoFisher Scientific Certificate of Analysis (CoA) Specialist Agent**.\n\n"
            f"- **Action Staged**: `{hitl_res.action_type}`\n"
            f"- **Catalog Number**: `{hitl_res.catalog_number}`\n"
            f"- **Lot Number**: `{hitl_res.lot_number or 'All Matching Lots'}`\n"
            f"- **Description**: {hitl_res.action_description}\n"
            f"- **Regulatory Compliance Warning**: {hitl_res.regulatory_impact_warning}\n\n"
            f"> **Next Step**: {hitl_res.approval_instructions}\n\n"
            f"🔗 **Portal Reference**: [{FISHERSCI_CERTIFICATES_PORTAL_URL}]({FISHERSCI_CERTIFICATES_PORTAL_URL})"
        )

    # 2. Check if the user is asking about supported Certificate Types without a catalog number
    if (
        any(kw in lowered for kw in ("certificate types", "types of certificates", "what certificate types"))
        and not resolved_catalog
    ):
        cert_types = list_supported_certificate_types()
        lines = [
            "# 📜 ThermoFisher / Fisher Scientific Supported Certificate Types",
            "",
            f"On the **ThermoFisher Scientific Documents & Certificates Portal** ([{FISHERSCI_CERTIFICATES_PORTAL_URL}]({FISHERSCI_CERTIFICATES_PORTAL_URL})), "
            f"the default selection is **`{cert_types.default_certificate_type}`**. You can search across the following certificate categories:",
            "",
            "| Certificate Type | Regulatory Description | Typical Products |",
            "| :--- | :--- | :--- |",
        ]
        for item in cert_types.supported_types:
            lines.append(
                f"| **{item.certificate_type}** | {item.description} | {item.typical_products} |"
            )
        lines.extend(
            [
                "",
                "### 🔍 Ready to Search?",
                "- Provide a **Catalog Number** (*Mandatory, minimum of 2 characters*, e.g., `AC29`, `AC299131000`, `A452-4`) and an optional **Lot Number** (e.g., `A0487095`).",
            ]
        )
        return "\n".join(lines)

    # 3. Validate Catalog Number (Mandatory, minimum 2 characters) & Optional Lot Number
    val_res = validate_catalog_and_lot_parameters(
        catalog_number=resolved_catalog or "",
        lot_number=extracted_lots[0] if len(extracted_lots) == 1 else None,
    )
    if val_res.status == "error" and val_res.error_recovery is not None:
        rec = val_res.error_recovery
        return (
            f"# ⚠️ Catalog Number Validation Error (`{rec.error_code}`)\n\n"
            f"I am the **ThermoFisher Scientific Certificate of Analysis (CoA) Specialist Agent**.\n\n"
            f"- **Validation Issue**: {val_res.validation_message}\n"
            f"- **Invalid Parameter**: `{rec.invalid_parameter}` (Provided: `{rec.provided_value or 'empty'}`)\n"
            f"- **Portal Requirement**: On [{FISHERSCI_CERTIFICATES_PORTAL_URL}]({FISHERSCI_CERTIFICATES_PORTAL_URL}), "
            f"**Catalog Number is mandatory and must be a minimum of 2 characters**, while **Lot Number is optional**.\n"
            f"- **Allowed Examples**: {', '.join(rec.allowed_values)}\n\n"
            f"### 💡 How to Proceed\n"
            f"{rec.llm_recovery_instruction}\n\n"
            f"**Try one of these valid queries:**\n"
            f"- `Search for Certificate of Analysis for catalog number AC29`\n"
            f"- `Find Certificate of Analysis for catalog number AC29 and lot number A0487095`\n"
            f"- `Search CoA for catalog A452-4`"
        )

    norm_cat = val_res.normalized_catalog_number
    norm_lot = extracted_lots[0] if len(extracted_lots) == 1 else None

    # 4. Execute Certificate of Analysis search
    search_out = search_certificates_of_analysis(
        catalog_number=norm_cat,
        lot_number=norm_lot,
        certificate_type="Certificate of Analysis",
        prefer_live_api=prefer_live_api,
    )

    sections: list[str] = [
        f"# 🧪 ThermoFisher Scientific — Certificate of Analysis Search Results (`{norm_cat}`)",
        "",
        f"Here are the **Certificate of Analysis** results retrieved from the ThermoFisher / Fisher Scientific "
        f"Documents & Certificates portal ([{FISHERSCI_CERTIFICATES_PORTAL_URL}]({FISHERSCI_CERTIFICATES_PORTAL_URL})):",
        "",
        f"- **Certificate Type**: `{search_out.certificate_type_queried}`",
        f"- **Catalog Number (Mandatory, ≥ 2 chars)**: `{norm_cat}` ({val_res.detected_brand})",
        f"- **Lot Number (Optional)**: `{norm_lot or 'Not specified (Showing all matching lots)'}`",
        f"- **Data Source**: `{search_out.data_source}` (`{search_out.total_results}` matching record(s))",
    ]

    if memory_context_summary and not extracted_cat and session_active_catalog:
        sections.extend(
            [
                "",
                f"> **Session Memory Active**: Automatically applied your active session Catalog Number (`{session_active_catalog}`) to this query.",
            ]
        )

    # 5. Render the exact 5-Column Fisher Scientific Results Table matching the screenshot
    sections.extend(
        [
            "",
            "## 📋 Search for Certificates — Results Table",
            "",
            "| Lot # | Certificate Type | Date | Catalog Number(s) | Product Name |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
    )

    if search_out.results:
        for row in search_out.results:
            sections.append(
                f"| **{row.lot_number}** | {row.certificate_type} | {row.date} | {row.catalog_numbers_display} | {row.product_name} |"
            )
    else:
        sections.append(
            f"| *No matching lots* | {search_out.certificate_type_queried} | — | `{norm_cat}` | *No Certificate of Analysis found for this Catalog/Lot combination* |"
        )

    # 6. Expand any 'View All' multi-SKU entries so the user has complete SKU visibility
    multi_sku_rows = [r for r in search_out.results if r.has_more_catalog_numbers]
    if multi_sku_rows:
        sections.extend(
            [
                "",
                "### 🔍 Expanded `View All` Catalog Number(s)",
            ]
        )
        for r in multi_sku_rows:
            sections.append(
                f"- **Lot `{r.lot_number}`** ({r.product_name}): `{', '.join(r.catalog_numbers)}`"
            )

    # 7. If a single specific Lot Number was queried or only 1 result matched, include deep lot verification details
    if len(extracted_lots) == 1 and search_out.results:
        lot_detail = get_certificate_lot_details(
            catalog_number=norm_cat,
            lot_number=extracted_lots[0],
            prefer_live_api=prefer_live_api,
        )
        if lot_detail.found and lot_detail.record:
            sections.extend(
                [
                    "",
                    f"## 🔬 Lot Verification Details (`{lot_detail.record.lot_number}`)",
                ]
            )
            for note in lot_detail.quality_compliance_notes:
                sections.append(f"- {note}")
            sections.append("")
            sections.append("### 📥 How to Download the Signed PDF on Fisher Scientific")
            for step in lot_detail.portal_verification_steps:
                sections.append(f"- {step}")

    # 8. If 2+ Lot Numbers were mentioned, include a side-by-side Lot Comparison
    if len(extracted_lots) >= 2:
        comp = compare_certificate_lots(
            catalog_number=norm_cat,
            lot_numbers=extracted_lots,
            prefer_live_api=prefer_live_api,
        )
        sections.extend(
            [
                "",
                f"## ⚖️ Multi-Lot Comparison (`{', '.join(extracted_lots)}`)",
                f"- **Summary**: {comp.comparison_summary}",
                f"- **Most Recent Lot**: `{comp.most_recent_lot or 'N/A'}`",
            ]
        )
        for rec in comp.matched_records:
            sections.append(
                f"  - **Lot `{rec.lot_number}`**: Published **{rec.date}** | SKUs: `{', '.join(rec.catalog_numbers)}` | Product: *{rec.product_name}*"
            )

    sections.extend(
        [
            "",
            "### 🔗 Official ThermoFisher / Fisher Scientific Portal",
            f"- **Search URL**: [{FISHERSCI_CERTIFICATES_PORTAL_URL}]({FISHERSCI_CERTIFICATES_PORTAL_URL})",
            "- **Input Rules**: Catalog Number (*Mandatory, min 2 characters*) | Lot Number (*Optional*)",
        ]
    )

    draft_text = "\n".join(sections)
    self_eval = guardrail_engine.self_evaluate_and_refine(
        sanitized_prompt,
        draft_text,
        is_search_results_response=True,
    )
    return self_eval.refined_response


async def arun_thermofisher_query(
    prompt: str,
    *,
    session_id: str = "default",
    model: str | None = None,
    prefer_live_api: bool = True,
    memory_manager: PersistentSessionMemoryManager | None = None,
) -> str:
    """Asynchronously executes a user query with multi-agent routing, persistent SQLite/vector memory, guardrails, and tracing."""
    start_perf = time.perf_counter()
    mem = memory_manager or session_memory_manager

    with tracer.start_as_current_span(
        "thermofisher.request.arun_thermofisher_query",
        component="orchestrator",
        attributes={"session_id": session_id, "prefer_live_api": prefer_live_api},
    ) as root_span:
        # 1. Input Guardrail & PII Scrubbing
        input_eval = guardrail_engine.evaluate_input(prompt)
        sanitized_prompt = input_eval.sanitized_prompt

        # 2. Async Memory Context Retrieval (before turn persistence so we can inspect active session catalog)
        context_window = await mem.abuild_context_window(session_id=session_id, current_query=sanitized_prompt)
        session_active_catalog = context_window.researcher_profile.last_active_catalog_number

        extracted_cat, extracted_lots = extract_catalog_and_lot_from_text(sanitized_prompt)
        effective_catalog = extracted_cat or session_active_catalog

        intent = intent_tracker.classify_intent(
            sanitized_prompt,
            session_id=session_id,
            extracted_catalog=effective_catalog,
            extracted_lots=extracted_lots,
            is_in_scope=input_eval.is_in_scope,
        )

        # 3. Strategic Model Routing
        has_valid_cat = bool(effective_catalog and len(effective_catalog.strip()) >= 2)
        routing = model_router.route_query(
            sanitized_prompt,
            lot_count=len(extracted_lots),
            has_valid_catalog=has_valid_cat,
        )
        selected_model = model or routing.selected_model
        root_span.set_attribute("thermofisher.selected_model", selected_model)
        root_span.set_attribute("thermofisher.intent_category", intent.intent_category)

        # Persist user turn
        await mem.aadd_turn(session_id=session_id, role="user", content=sanitized_prompt)

        if not input_eval.allowed:
            latency_ms = (time.perf_counter() - start_perf) * 1000.0
            outcome_status: Any = (
                "GUARDRAIL_BLOCKED_SAFETY"
                if input_eval.block_reason == "PROMPT_INJECTION"
                else "GUARDRAIL_BLOCKED_SCOPE"
            )
            await mem.aadd_turn(session_id=session_id, role="assistant", content=input_eval.guardrail_message)
            intent_tracker.record_outcome(
                intent,
                actual_tools_invoked=["check_thermofisher_coa_scope"],
                sub_agents_dispatched=["safety_and_critic_specialist"],
                selected_model=selected_model,
                outcome_status=outcome_status,
                self_eval_score=1.0,
                latency_ms=latency_ms,
            )
            return input_eval.guardrail_message

        # 4. Resolve API Key from Google Cloud Secret Manager (with local fallback)
        api_key, secret_meta = secret_manager_service.get_gemini_api_key()
        root_span.set_attribute("thermofisher.secret_source", secret_meta.source)

        response_text: str | None = None
        if (
            prefer_live_api
            and api_key
            and not intent.requires_hitl_confirmation
            and intent.intent_category != "INVALID_CATALOG_NUMBER"
        ):
            try:
                from google import genai
                from google.genai import types

                augmented_prompt = (
                    f"{context_window.formatted_context_prompt}\n\n### Current User Request\n{sanitized_prompt}"
                    if context_window.formatted_context_prompt
                    else sanitized_prompt
                )

                def _call_gemini(target_model: str) -> str | None:
                    client = genai.Client(api_key=api_key)
                    resp = client.models.generate_content(
                        model=target_model,
                        contents=augmented_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=THERMOFISHER_SYSTEM_PROMPT,
                            tools=ALL_THERMOFISHER_TOOLS,
                            safety_settings=get_cloud_native_safety_settings(),
                            temperature=0.2,
                        ),
                    )
                    return resp.text if resp and resp.text else None

                for candidate_model in [selected_model, *routing.fallback_models]:
                    try:
                        response_text = await asyncio.to_thread(_call_gemini, candidate_model)
                        if response_text:
                            selected_model = candidate_model
                            break
                    except Exception as model_exc:
                        structured_logger.warning(
                            f"Model {candidate_model} invocation failed, trying next fallback: {model_exc}",
                            component="orchestrator",
                            event_type="model_fallback",
                            session_id=session_id,
                        )
            except Exception:
                response_text = None

        if not response_text:
            response_text = await asyncio.to_thread(
                render_tool_assisted_response,
                sanitized_prompt,
                session_active_catalog=session_active_catalog,
                memory_context_summary=context_window.formatted_context_prompt,
                prefer_live_api=prefer_live_api,
            )

        # 5. Agentic Self-Evaluation & Auto-Refinement Critic Pass
        is_search_resp = intent.intent_category in {
            "COA_CATALOG_SEARCH",
            "COA_CATALOG_AND_LOT_SEARCH",
            "LOT_COMPARISON",
        }
        self_eval = guardrail_engine.self_evaluate_and_refine(
            sanitized_prompt,
            response_text,
            is_search_results_response=is_search_resp,
        )
        final_response = self_eval.refined_response

        # 6. Persist Assistant Turn & Record Intent-vs-Outcome Telemetry
        await mem.aadd_turn(session_id=session_id, role="assistant", content=final_response)
        latency_ms = (time.perf_counter() - start_perf) * 1000.0

        if intent.requires_hitl_confirmation or "Human-in-the-Loop QA Confirmation Required" in final_response:
            final_status: Any = "AWAITING_HITL_CONFIRMATION"
        elif intent.intent_category == "INVALID_CATALOG_NUMBER" or "Catalog Number Validation Error" in final_response:
            final_status = "VALIDATION_GUIDANCE_RETURNED"
        else:
            final_status = "FULFILLED"

        intent_tracker.record_outcome(
            intent,
            actual_tools_invoked=intent.expected_tools,
            sub_agents_dispatched=[
                "coa_search_specialist",
                "lot_verification_and_comparison_specialist",
                "regulatory_and_hitl_specialist",
                "safety_and_critic_specialist",
            ],
            selected_model=selected_model,
            outcome_status=final_status,
            self_eval_score=self_eval.overall_score,
            latency_ms=latency_ms,
        )
        return final_response


def run_thermofisher_query(
    prompt: str,
    model: str | None = None,
    prefer_live_api: bool = True,
    session_id: str = "default",
) -> str:
    """Synchronous entrypoint wrapping `arun_thermofisher_query` for CLI and standard script compatibility."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                arun_thermofisher_query(
                    prompt,
                    session_id=session_id,
                    model=model,
                    prefer_live_api=prefer_live_api,
                ),
            )
            return future.result()

    return asyncio.run(
        arun_thermofisher_query(
            prompt,
            session_id=session_id,
            model=model,
            prefer_live_api=prefer_live_api,
        )
    )


__all__ = [
    "THERMOFISHER_ROLE_HEADER",
    "THERMOFISHER_SYSTEM_PROMPT",
    "arun_thermofisher_query",
    "create_thermofisher_agent",
    "render_tool_assisted_response",
    "root_agent",
    "run_thermofisher_query",
]
