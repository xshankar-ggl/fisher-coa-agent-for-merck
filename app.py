"""FastAPI Web UI and REST API for the ThermoFisher Scientific Certificate of Analysis (CoA) Specialist Agent."""

from __future__ import annotations

from typing import Optional
import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from thermofisher_agent.agent import THERMOFISHER_ROLE_HEADER, arun_thermofisher_query
from thermofisher_agent.hitl import ConfirmationDecisionOutput, hitl_manager
from thermofisher_agent.memory import CompactedHistorySummary, SessionStateModel, session_memory_manager
from thermofisher_agent.observability import intent_tracker, pii_redactor, structured_logger, tracer
from thermofisher_agent.schemas import (
    CatalogValidationOutput,
    CertificateSearchOutput,
    HighStakesRegulatoryOutput,
    LotComparisonOutput,
    LotDetailAnalysisOutput,
    SupportedCertificateTypesOutput,
)
from thermofisher_agent.tools import (
    compare_certificate_lots,
    get_certificate_lot_details,
    list_supported_certificate_types,
    request_regulatory_action_confirmation,
    resolve_regulatory_action_confirmation,
    search_certificates_of_analysis,
    validate_catalog_and_lot_parameters,
)

app = FastAPI(
    title="ThermoFisher Scientific — Certificate of Analysis (CoA) Specialist Agent",
    description=THERMOFISHER_ROLE_HEADER,
    version="2.0.0",
)


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Natural-language Certificate of Analysis prompt.")
    session_id: str = Field(default="default", description="Persistent conversation session ID.")


class ChatResponse(BaseModel):
    prompt: str
    session_id: str
    response: str


class HealthResponse(BaseModel):
    status: str
    agent: str
    specialty: str
    portal_url: str
    version: str


class CertificateSearchRequest(BaseModel):
    catalog_number: str = Field(
        ...,
        description="Mandatory Catalog Number (minimum of 2 characters, e.g., 'AC29').",
    )
    lot_number: Optional[str] = Field(
        default=None,
        description="Optional Lot Number (e.g., 'A0487095').",
    )
    certificate_type: str = Field(
        default="Certificate of Analysis",
        description="Certificate type filter on Fisher Scientific.",
    )
    page_no: int = Field(default=1, ge=1)
    results_per_page: int = Field(default=10, ge=1, le=60)


class LotCompareRequest(BaseModel):
    catalog_number: str = Field(..., description="Mandatory Catalog Number (minimum 2 characters).")
    lot_numbers: list[str] = Field(
        default_factory=lambda: ["A0487095", "A0487009"],
        description="Two or more Lot Numbers to compare.",
    )


class HITLStageRequest(BaseModel):
    action_type: str = Field(default="LIMS_QUALITY_RELEASE_APPROVAL")
    catalog_number: str = Field(..., description="Mandatory Catalog Number (min 2 chars).")
    lot_number: Optional[str] = Field(default=None)
    action_description: str = Field(..., min_length=3)
    gxp_regulated: bool = Field(default=True)


class HITLConfirmRequest(BaseModel):
    confirmation_id: str = Field(..., min_length=3)
    approved: bool = Field(...)
    reviewer_id: str = Field(default="qa_specialist")
    reviewer_note: str = Field(default="")


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        agent="ThermoFisher Scientific CoA Specialist Agent",
        specialty="Certificate of Analysis Search & Regulatory Verification",
        portal_url="https://www.fishersci.com/search/documents?type=certificates",
        version="2.0.0",
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_agent(req: ChatRequest) -> ChatResponse:
    reply = await arun_thermofisher_query(req.prompt, session_id=req.session_id)
    return ChatResponse(prompt=req.prompt, session_id=req.session_id, response=reply)


@app.post("/api/certificates/search", response_model=CertificateSearchOutput)
async def search_certificates_endpoint(req: CertificateSearchRequest) -> CertificateSearchOutput:
    return search_certificates_of_analysis(
        catalog_number=req.catalog_number,
        lot_number=req.lot_number,
        certificate_type=req.certificate_type,
        page_no=req.page_no,
        results_per_page=req.results_per_page,
    )


@app.get("/api/certificates/validate", response_model=CatalogValidationOutput)
async def validate_parameters_endpoint(
    catalog_number: str = Query(..., description="Mandatory Catalog Number (min 2 chars)"),
    lot_number: Optional[str] = Query(None, description="Optional Lot Number"),
) -> CatalogValidationOutput:
    return validate_catalog_and_lot_parameters(catalog_number=catalog_number, lot_number=lot_number)


@app.get("/api/certificates/lot-details", response_model=LotDetailAnalysisOutput)
async def lot_details_endpoint(
    catalog_number: str = Query("AC29"),
    lot_number: str = Query("A0487095"),
) -> LotDetailAnalysisOutput:
    return get_certificate_lot_details(catalog_number=catalog_number, lot_number=lot_number)


@app.post("/api/certificates/compare-lots", response_model=LotComparisonOutput)
async def compare_lots_endpoint(req: LotCompareRequest) -> LotComparisonOutput:
    return compare_certificate_lots(catalog_number=req.catalog_number, lot_numbers=req.lot_numbers)


@app.get("/api/certificates/types", response_model=SupportedCertificateTypesOutput)
async def certificate_types_endpoint() -> SupportedCertificateTypesOutput:
    return list_supported_certificate_types()


@app.get("/api/session/{session_id}", response_model=SessionStateModel)
async def get_session_endpoint(session_id: str) -> SessionStateModel:
    return await session_memory_manager.aget_session_state(session_id=session_id)


@app.post("/api/session/{session_id}/compact", response_model=CompactedHistorySummary)
async def compact_session_endpoint(session_id: str) -> CompactedHistorySummary:
    return await session_memory_manager.acompact_session_history(session_id=session_id, force=True)


@app.post("/api/hitl/stage", response_model=HighStakesRegulatoryOutput)
async def stage_hitl_action(req: HITLStageRequest) -> HighStakesRegulatoryOutput:
    return request_regulatory_action_confirmation(
        action_type=req.action_type,
        catalog_number=req.catalog_number,
        lot_number=req.lot_number,
        action_description=req.action_description,
        gxp_regulated=req.gxp_regulated,
    )


@app.post("/api/hitl/confirm", response_model=ConfirmationDecisionOutput)
async def confirm_hitl_action(req: HITLConfirmRequest) -> ConfirmationDecisionOutput:
    return resolve_regulatory_action_confirmation(
        confirmation_id=req.confirmation_id,
        approved=req.approved,
        reviewer_id=req.reviewer_id,
        reviewer_note=req.reviewer_note,
    )


@app.get("/api/observability/metrics")
async def observability_metrics_endpoint() -> dict[str, object]:
    return {
        "intent_vs_outcome": intent_tracker.get_metrics_summary(),
        "total_pii_redactions": pii_redactor.total_redactions,
        "recorded_spans_count": len(tracer.completed_spans),
        "structured_logs_count": len(structured_logger.entries),
        "pending_hitl_actions": [p.model_dump() for p in hitl_manager.list_pending_actions()],
    }


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ThermoFisher Scientific — Search for Certificates & AI Chat Agent</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    .prose h1 { font-size: 1.35rem; font-weight: 700; margin-top: 0.75rem; margin-bottom: 0.5rem; color: #004b87; }
    .prose h2 { font-size: 1.15rem; font-weight: 700; margin-top: 1rem; margin-bottom: 0.5rem; color: #1e293b; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.25rem; }
    .prose h3 { font-size: 1rem; font-weight: 600; margin-top: 0.75rem; margin-bottom: 0.35rem; color: #005dab; }
    .prose ul { list-style-type: disc; padding-left: 1.4rem; margin-bottom: 0.75rem; }
    .prose li { margin-bottom: 0.25rem; }
    .prose p { margin-bottom: 0.65rem; line-height: 1.55; }
    .prose table { width: 100%; border-collapse: collapse; margin: 0.75rem 0; font-size: 0.82rem; }
    .prose th { text-align: left; padding: 0.6rem 0.5rem; border-bottom: 2px solid #334155; font-weight: 700; color: #1e293b; }
    .prose td { padding: 0.6rem 0.5rem; border-bottom: 1px solid #e2e8f0; color: #334155; }
    .prose blockquote { border-left: 4px solid #005dab; background: #f0f7ff; padding: 0.5rem 0.85rem; border-radius: 0.375rem; margin: 0.65rem 0; }
    .prose code { background: #f1f5f9; padding: 0.12rem 0.35rem; border-radius: 0.25rem; font-size: 0.82rem; }
  </style>
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen">
  <!-- Top Fisher Scientific Navy Banner -->
  <div class="bg-[#003865] text-white text-xs py-2 px-6 flex items-center justify-between">
    <div class="flex items-center gap-6 font-medium">
      <span>Fisher Healthcare</span>
      <span>Fisher Science Education</span>
      <span class="text-sky-200">Documents &amp; Certificates Portal: https://www.fishersci.com/search/documents?type=certificates</span>
    </div>
    <div class="flex items-center gap-2">
      <span class="inline-block w-2 h-2 rounded-full bg-emerald-400"></span>
      <span>5-Axis Enterprise Agent (Google ADK + Gemini + OpenTelemetry)</span>
    </div>
  </div>

  <!-- Main Fisher Scientific Header -->
  <header class="bg-white border-b border-slate-200 shadow-sm">
    <div class="max-w-7xl mx-auto px-6 py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
      <div class="flex items-center gap-4">
        <div class="w-10 h-10 rounded-full border-2 border-[#005dab] flex items-center justify-center text-[#005dab] font-bold text-xl">f</div>
        <div>
          <div class="text-2xl font-light tracking-tight text-slate-800">
            <span class="font-semibold text-[#005dab]">fisher</span> scientific
          </div>
          <div class="text-[11px] text-slate-500 -mt-1">part of Thermo Fisher Scientific • Certificate of Analysis AI Agent</div>
        </div>
      </div>
      <div class="flex flex-wrap gap-2 text-xs text-slate-600">
        <span class="px-3 py-1.5 rounded-md bg-slate-100 font-medium">Catalog Number: Mandatory (Min 2 chars)</span>
        <span class="px-3 py-1.5 rounded-md bg-slate-100 font-medium">Lot Number: Optional</span>
      </div>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
    <!-- Left 7 Columns: Exact Fisher Scientific "Search for certificates" Portal Interface -->
    <section class="lg:col-span-7 space-y-6">
      <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <h1 class="text-2xl font-bold text-slate-900 mb-5">Search for certificates</h1>

        <!-- Search Form matching Fisher Scientific Screenshot -->
        <div class="grid grid-cols-1 md:grid-cols-12 gap-3 items-start">
          <div class="md:col-span-4">
            <select id="certTypeSelect" class="w-full h-11 rounded border border-slate-400 px-3 text-sm text-slate-800 bg-white focus:border-[#005dab] focus:outline-none">
              <option value="Certificate of Analysis" selected>Certificate of Analysis</option>
              <option value="All Certificates">All Certificates</option>
              <option value="Certificate of Compliance / Conformity / Conformance">Certificate of Compliance / Conformity</option>
              <option value="Certificate of Sterility">Certificate of Sterility</option>
              <option value="Certificate of Validation">Certificate of Validation</option>
              <option value="Certificate of Animal Origin">Certificate of Animal Origin</option>
              <option value="Certificate of Guarantee">Certificate of Guarantee</option>
              <option value="REACH">REACH</option>
              <option value="Letter of Certification">Letter of Certification</option>
            </select>
          </div>
          <div class="md:col-span-3">
            <input id="catalogInput" type="text" value="AC29" placeholder="Catalog Number (Min 2 chars)"
              class="w-full h-11 rounded border border-slate-400 px-3 text-sm text-slate-800 focus:border-[#005dab] focus:outline-none" />
          </div>
          <div class="md:col-span-3">
            <input id="lotInput" type="text" value="" placeholder="Lot number (Optional)"
              class="w-full h-11 rounded border border-slate-400 px-3 text-sm text-slate-800 focus:border-[#005dab] focus:outline-none" />
          </div>
          <div class="md:col-span-2">
            <button onclick="runDirectSearch()"
              class="w-full h-11 rounded border border-slate-400 bg-white hover:bg-slate-50 text-[#005dab] font-semibold text-sm transition shadow-sm">
              Search
            </button>
          </div>
        </div>

        <!-- Validation Banner -->
        <div id="validationAlert" class="hidden mt-4 p-3 rounded-lg bg-red-50 border border-red-200 text-red-800 text-xs"></div>

        <!-- 5-Column Results Table matching Fisher Scientific Screenshot -->
        <div class="mt-6 overflow-x-auto">
          <table class="w-full text-left border-collapse">
            <thead>
              <tr class="border-b-2 border-slate-700 text-xs font-bold text-slate-800">
                <th class="py-3 px-2">Lot #</th>
                <th class="py-3 px-2">Certificate Type</th>
                <th class="py-3 px-2">Date</th>
                <th class="py-3 px-2">Catalog Number(s)</th>
                <th class="py-3 px-2">Product Name</th>
              </tr>
            </thead>
            <tbody id="resultsTableBody" class="divide-y divide-slate-200 text-xs text-slate-700">
            </tbody>
          </table>
        </div>
        <div id="tableFooterMeta" class="mt-3 text-[11px] text-slate-400 flex justify-between"></div>
      </div>

      <!-- Lot Details Drawer -->
      <div id="lotDetailCard" class="hidden bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <div class="flex items-center justify-between border-b border-slate-100 pb-3 mb-3">
          <h3 class="font-bold text-slate-900 text-sm">🔬 Selected Certificate of Analysis Lot Verification</h3>
          <button onclick="document.getElementById('lotDetailCard').classList.add('hidden')" class="text-xs text-slate-400 hover:text-slate-600">Close ✕</button>
        </div>
        <div id="lotDetailContent" class="text-xs text-slate-700 space-y-2"></div>
      </div>
    </section>

    <!-- Right 5 Columns: Conversational AI Agent + 5-Axis Telemetry -->
    <aside class="lg:col-span-5 space-y-6">
      <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <div class="flex items-center justify-between">
          <h2 class="text-base font-bold text-slate-900 flex items-center gap-2">
            <span>💬</span> ThermoFisher CoA Chat Agent
          </h2>
          <span id="chatStatus" class="text-xs text-slate-400">Ready</span>
        </div>
        <p class="text-xs text-slate-500 mt-1">
          Ask in natural language to search by Catalog Number (mandatory, ≥2 chars) and optional Lot Number.
        </p>

        <!-- Preset Prompts -->
        <div class="flex flex-wrap gap-1.5 mt-3">
          <button onclick="setChatPreset('Search for Certificate of Analysis for catalog number AC29')"
            class="text-[11px] bg-blue-50 hover:bg-blue-100 text-[#005dab] border border-blue-200 px-2.5 py-1 rounded-full transition">
            🔍 Catalog AC29 (All Lots)
          </button>
          <button onclick="setChatPreset('Search for Certificate of Analysis for catalog number AC29 and lot number A0487095')"
            class="text-[11px] bg-blue-50 hover:bg-blue-100 text-[#005dab] border border-blue-200 px-2.5 py-1 rounded-full transition">
            🧪 Catalog AC29 + Lot A0487095
          </button>
          <button onclick="setChatPreset('Compare lots A0487095 and A0487009 for catalog number AC29')"
            class="text-[11px] bg-indigo-50 hover:bg-indigo-100 text-indigo-800 border border-indigo-200 px-2.5 py-1 rounded-full transition">
            ⚖️ Compare Lots A0487095 vs A0487009
          </button>
          <button onclick="setChatPreset('Search for Certificate of Analysis for catalog number A')"
            class="text-[11px] bg-amber-50 hover:bg-amber-100 text-amber-800 border border-amber-200 px-2.5 py-1 rounded-full transition">
            ⚠️ Test &lt;2 Char Catalog Validation ('A')
          </button>
          <button onclick="setChatPreset('Approve LIMS release for catalog AC299131000 lot A0487095')"
            class="text-[11px] bg-rose-50 hover:bg-rose-100 text-rose-800 border border-rose-200 px-2.5 py-1 rounded-full transition">
            🛑 Test HITL LIMS Release Gate
          </button>
        </div>

        <div class="mt-3">
          <textarea id="chatPrompt" rows="3"
            class="w-full rounded-lg border border-slate-300 p-3 text-xs focus:ring-2 focus:ring-[#005dab] focus:outline-none"
            placeholder="e.g., Search for Certificate of Analysis for catalog number AC29 and lot number A0487095..."></textarea>
          <div class="mt-2 flex justify-end">
            <button onclick="sendChat()"
              class="bg-[#005dab] hover:bg-[#004b87] text-white font-semibold text-xs px-4 py-2 rounded-lg shadow-sm transition">
              Send to CoA Agent ✨
            </button>
          </div>
        </div>

        <div class="mt-4 pt-4 border-t border-slate-100">
          <div id="chatOutput" class="prose max-w-none text-xs text-slate-700 max-h-[420px] overflow-y-auto">
            <p class="text-slate-400 italic">Select a preset above or type a question to interact with the ThermoFisher CoA Chat Agent.</p>
          </div>
        </div>
      </div>

      <!-- Observability & Telemetry Card -->
      <div class="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
        <div class="flex items-center justify-between">
          <h3 class="text-xs font-bold uppercase tracking-wider text-slate-700">📊 5-Axis Observability &amp; Tracing</h3>
          <button onclick="refreshMetrics()" class="text-xs text-[#005dab] hover:underline">Refresh</button>
        </div>
        <div id="metricsBox" class="mt-3 text-xs text-slate-600 space-y-1">
          Loading telemetry...
        </div>
      </div>
    </aside>
  </main>

  <script>
    function toggleSkus(idx, fullText, shortText) {
      const span = document.getElementById('sku-text-' + idx);
      const btn = document.getElementById('sku-btn-' + idx);
      if (btn.textContent === 'View All') {
        span.textContent = fullText + ' ';
        btn.textContent = 'View Less';
      } else {
        span.textContent = shortText + ' ';
        btn.textContent = 'View All';
      }
    }

    async function runDirectSearch() {
      const catalog_number = document.getElementById('catalogInput').value.trim();
      const lot_number_raw = document.getElementById('lotInput').value.trim();
      const certificate_type = document.getElementById('certTypeSelect').value;
      const alertBox = document.getElementById('validationAlert');
      const tbody = document.getElementById('resultsTableBody');
      const footer = document.getElementById('tableFooterMeta');

      alertBox.classList.add('hidden');
      tbody.innerHTML = '<tr><td colspan="5" class="py-6 text-center text-slate-400">Searching Fisher Scientific Certificates...</td></tr>';

      const res = await fetch('/api/certificates/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          catalog_number,
          lot_number: lot_number_raw || null,
          certificate_type
        })
      });
      const data = await res.json();

      if (data.status === 'error') {
        alertBox.classList.remove('hidden');
        const rec = data.error_recovery || {};
        alertBox.innerHTML = `<strong>Validation Error (${rec.error_code || 'INVALID_INPUT'}):</strong> ${data.summary}<br/><span class="text-red-600">${rec.llm_recovery_instruction || ''}</span>`;
        tbody.innerHTML = '<tr><td colspan="5" class="py-6 text-center text-red-500">Please enter a Catalog Number with a minimum of 2 characters.</td></tr>';
        footer.textContent = '';
        refreshMetrics();
        return;
      }

      if (!data.results || data.results.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="py-6 text-center text-slate-400">No certificates found matching your criteria.</td></tr>';
        footer.textContent = `Source: ${data.data_source} | 0 results`;
        refreshMetrics();
        return;
      }

      tbody.innerHTML = '';
      data.results.forEach((row, idx) => {
        let skuCellHtml = '';
        if (row.has_more_catalog_numbers) {
          const shortPrefix = row.catalog_numbers_display.replace(' View All', '');
          const fullList = row.catalog_numbers.join(', ');
          skuCellHtml = `<span id="sku-text-${idx}">${shortPrefix} </span><a id="sku-btn-${idx}" href="javascript:void(0)" onclick="toggleSkus(${idx}, '${fullList}', '${shortPrefix}')" class="text-[#005dab] underline font-medium">View All</a>`;
        } else {
          skuCellHtml = row.catalog_numbers_display;
        }

        const tr = document.createElement('tr');
        tr.className = 'hover:bg-slate-50 transition';
        tr.innerHTML = `
          <td class="py-3.5 px-2 font-medium">
            <a href="javascript:void(0)" onclick="inspectLot('${data.catalog_number_queried}', '${row.lot_number}')" class="text-[#005dab] hover:underline">${row.lot_number}</a>
          </td>
          <td class="py-3.5 px-2 text-slate-600">${row.certificate_type}</td>
          <td class="py-3.5 px-2 text-slate-600 whitespace-nowrap">${row.date}</td>
          <td class="py-3.5 px-2 text-slate-600">${skuCellHtml}</td>
          <td class="py-3.5 px-2 text-slate-700">${row.product_name}</td>
        `;
        tbody.appendChild(tr);
      });

      footer.innerHTML = `<span>Data Source: <strong>${data.data_source}</strong></span><span>Showing ${data.results.length} certificate(s) for Catalog <strong>${data.catalog_number_queried}</strong></span>`;
      refreshMetrics();
    }

    async function inspectLot(catalogNo, lotNo) {
      const card = document.getElementById('lotDetailCard');
      const content = document.getElementById('lotDetailContent');
      card.classList.remove('hidden');
      content.innerHTML = 'Loading lot verification metadata...';
      const res = await fetch(`/api/certificates/lot-details?catalog_number=${encodeURIComponent(catalogNo)}&lot_number=${encodeURIComponent(lotNo)}`);
      const data = await res.json();
      if (!data.found || !data.record) {
        content.innerHTML = '<p class="text-red-600">Lot details not found.</p>';
        return;
      }
      const r = data.record;
      content.innerHTML = `
        <div class="grid grid-cols-2 gap-2 bg-slate-50 p-3 rounded border border-slate-200">
          <div><strong>Lot #:</strong> <span class="text-[#005dab] font-semibold">${r.lot_number}</span></div>
          <div><strong>Published Date:</strong> ${r.date}</div>
          <div class="col-span-2"><strong>Product Name:</strong> ${r.product_name}</div>
          <div class="col-span-2"><strong>All Covered Catalog Number(s):</strong> <code>${r.catalog_numbers.join(', ')}</code></div>
          <div class="col-span-2"><strong>eCertificate Asset Path:</strong> <code>${r.ecertificate_path}</code></div>
        </div>
        <div class="pt-2">
          <div class="font-semibold text-slate-800 mb-1">Portal Verification Steps:</div>
          <ul class="list-disc pl-4 space-y-0.5 text-slate-600">
            ${data.portal_verification_steps.map(s => `<li>${s}</li>`).join('')}
          </ul>
        </div>
      `;
      refreshMetrics();
    }

    function setChatPreset(text) {
      document.getElementById('chatPrompt').value = text;
      sendChat();
    }

    async function sendChat() {
      const prompt = document.getElementById('chatPrompt').value.trim();
      if (!prompt) return;
      const status = document.getElementById('chatStatus');
      const out = document.getElementById('chatOutput');
      status.textContent = 'Querying agent...';
      out.innerHTML = '<p class="text-slate-400 animate-pulse">Validating Catalog/Lot inputs and querying Fisher Scientific Certificates...</p>';
      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt })
        });
        const data = await res.json();
        out.innerHTML = marked.parse(data.response);
        status.textContent = 'Complete';
        refreshMetrics();
      } catch (e) {
        out.innerHTML = '<p class="text-red-600">Error contacting ThermoFisher CoA Agent.</p>';
        status.textContent = 'Error';
      }
    }

    async function refreshMetrics() {
      try {
        const res = await fetch('/api/observability/metrics');
        const data = await res.json();
        const ivo = data.intent_vs_outcome || {};
        document.getElementById('metricsBox').innerHTML = `
          <div class="grid grid-cols-2 gap-2">
            <div class="bg-slate-50 p-2 rounded border border-slate-100"><strong>OpenTelemetry Spans:</strong> ${data.recorded_spans_count}</div>
            <div class="bg-slate-50 p-2 rounded border border-slate-100"><strong>Structured Logs:</strong> ${data.structured_logs_count}</div>
            <div class="bg-slate-50 p-2 rounded border border-slate-100"><strong>Goal Achievement:</strong> ${(ivo.goal_achievement_rate * 100).toFixed(0)}%</div>
            <div class="bg-slate-50 p-2 rounded border border-slate-100"><strong>Self-Eval Score:</strong> ${ivo.average_self_eval_score}</div>
          </div>
        `;
      } catch (e) {}
    }

    // Load initial AC29 table on startup (matching screenshot)
    window.addEventListener('DOMContentLoaded', () => {
      runDirectSearch();
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def serve_ui() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
