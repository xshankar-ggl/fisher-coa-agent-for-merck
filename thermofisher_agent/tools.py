"""Specialist tools for the ThermoFisher Scientific Certificate of Analysis (CoA) Agent.

Connects to Fisher Scientific's Documents & Certificates Portal (`https://www.fishersci.com/search/documents?type=certificates`)
and its underlying REST API (`https://www.fishersci.com/api/store/Assets/Documents/Certificates/v2/search`),
with strict Pydantic v2 output schemas, mandatory Catalog Number (>= 2 characters) validation,
optional Lot Number filtering, and guided LLM error recovery.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any

from thermofisher_agent.hitl import (
    ConfirmationDecisionOutput,
    hitl_manager,
    require_human_confirmation,
)
from thermofisher_agent.knowledge import (
    BRAND_PREFIX_MAP,
    CURATED_CERTIFICATE_ASSETS,
    FISHERSCI_CERTIFICATES_API_URL,
    FISHERSCI_CERTIFICATES_PORTAL_URL,
    SUPPORTED_CERTIFICATE_TYPES,
)
from thermofisher_agent.observability import structured_logger, trace_span
from thermofisher_agent.schemas import (
    CatalogValidationOutput,
    CertificateRecordSchema,
    CertificateSearchOutput,
    CertificateTypeInfoSchema,
    DomainScopeOutput,
    ErrorRecoveryGuidance,
    HighStakesRegulatoryOutput,
    LotComparisonOutput,
    LotDetailAnalysisOutput,
    SupportedCertificateTypesOutput,
)

# Valid characters for Catalog Number and Lot Number inputs
_VALID_CODE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-_./\s]*$")

# Out-of-scope topics unrelated to ThermoFisher / Fisher Scientific Certificates & Laboratory Products
OUT_OF_SCOPE_TOPICS: dict[str, str] = {
    "flight": "airline / flight booking",
    "hotel": "hotel / hospitality reservation",
    "itinerary": "vacation travel itinerary",
    "jr pass": "Japan Rail Pass transit planning",
    "ryokan": "Japanese inn booking",
    "stock price": "financial stock trading",
    "crypto": "cryptocurrency trading",
    "bitcoin": "cryptocurrency trading",
    "movie": "entertainment / movie recommendations",
    "recipe": "culinary cooking recipes",
    "weather forecast": "meteorological weather forecasting",
}

ALLOWED_CERTIFICATE_TYPE_NAMES: list[str] = [item["certificate_type"] for item in SUPPORTED_CERTIFICATE_TYPES]


def _infer_brand_from_catalog(catalog_number: str) -> str:
    """Infers the ThermoFisher / Fisher Scientific division from the catalog number prefix."""
    upper = catalog_number.strip().upper()
    for prefix in sorted(BRAND_PREFIX_MAP.keys(), key=len, reverse=True):
        if upper.startswith(prefix):
            return BRAND_PREFIX_MAP[prefix]
    return "ThermoFisher Scientific / Fisher Scientific Catalog"


def _format_publish_date(iso_str: str) -> str:
    """Formats an ISO-8601 timestamp (e.g. '2026-10-20T07:01:11Z') as 'Oct 20, 2026' matching fishersci.com."""
    if not iso_str:
        return "Unknown Date"
    try:
        cleaned = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        return f"{dt.strftime('%b')} {dt.day}, {dt.year}"
    except Exception:
        return iso_str[:10]


def _format_catalog_numbers_display(skus: list[str]) -> tuple[str, bool]:
    """Formats SKU list to match Fisher Scientific's 'Catalog Number(s)' column (with 'View All' when > 2 SKUs)."""
    if not skus:
        return ("N/A", False)
    if len(skus) <= 2:
        return (", ".join(skus), False)
    preview = f"{skus[0]}, {skus[1][:8]}... View All"
    return (preview, True)


def _asset_dict_to_schema(asset: dict[str, Any], queried_catalog: str) -> CertificateRecordSchema:
    """Converts a raw Fisher Scientific API asset dictionary into a strict `CertificateRecordSchema`."""
    skus = [str(s) for s in asset.get("sku", []) if s]
    display_skus, has_more = _format_catalog_numbers_display(skus)
    publish_iso = str(asset.get("publishDate", ""))
    formatted_date = _format_publish_date(publish_iso)
    titles = [str(t) for t in asset.get("productTitles", []) if t]
    product_name = titles[0] if titles else "ThermoFisher Scientific Certified Product"
    lot_no = str(asset.get("lotNumber") or asset.get("title") or "UNKNOWN")
    doc_type = str(asset.get("documentType") or "Certificate of Analysis")
    doc_id = str(asset.get("id") or f"cert-{lot_no.lower()}")
    path = str(asset.get("path") or f"api/ecertificate/{doc_id}")

    return CertificateRecordSchema(
        document_id=doc_id,
        lot_number=lot_no,
        certificate_type=doc_type,
        date=formatted_date,
        publish_date_iso=publish_iso,
        catalog_numbers=skus,
        catalog_numbers_display=display_skus,
        has_more_catalog_numbers=has_more,
        product_name=product_name,
        mpn_list=[str(m) for m in asset.get("mpn", []) if m],
        root_sku=[str(r) for r in asset.get("rootSku", []) if r],
        erp=str(asset.get("erp") or "MF_US"),
        languages=[str(lang) for lang in asset.get("languages", ["en"])],
        ecertificate_path=path,
        fishersci_search_url=FISHERSCI_CERTIFICATES_PORTAL_URL,
    )


def _fetch_live_fishersci_certificates(
    *,
    catalog_number: str,
    lot_number: str | None,
    certificate_type: str,
    page_no: int,
    results_per_page: int,
    partial_sku_search: bool,
    timeout_sec: float = 6.0,
) -> list[dict[str, Any]] | None:
    """Queries the live Fisher Scientific Certificates v2 REST API.

    Returns a list of raw asset dicts on success, or `None` if network/API is unreachable so the caller
    can transparently fall back to the curated Fisher Scientific catalog.
    """
    if os.getenv("THERMOFISHER_DISABLE_LIVE_HTTP", "").lower() in {"1", "true", "yes"}:
        return None

    params: dict[str, str] = {
        "skus": catalog_number,
        "targetSite": "FSUS",
        "partialLotNumber": "false",
        "country": "us",
        "sortBy": "publishDate",
        "sortType": "DESC",
        "pageNo": str(page_no),
        "rpp": str(results_per_page),
        "partialSkuSearch": "true" if partial_sku_search else "false",
    }
    if lot_number:
        params["lotNumbers"] = lot_number
    if certificate_type and certificate_type != "All Certificates":
        params["certificateType"] = certificate_type

    query_string = urllib.parse.urlencode(params)
    url = f"{FISHERSCI_CERTIFICATES_API_URL}?{query_string}"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ThermoFisherCoAAgent/2.0",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            asset_types = payload.get("assetTypes") or []
            for at in asset_types:
                if at.get("value") == "Certificates":
                    assets: list[dict[str, Any]] = []
                    for dt in at.get("documentTypes") or []:
                        assets.extend(dt.get("assets") or [])
                    return assets
            return []
    except Exception as exc:
        structured_logger.warning(
            f"Live FisherSci API call failed or timed out, using curated fallback catalog: {exc}",
            component="tools",
            event_type="fishersci_api_fallback",
            attributes={"catalog_number": catalog_number, "lot_number": lot_number},
        )
        return None


def _search_curated_catalog(
    *,
    catalog_number: str,
    lot_number: str | None,
    certificate_type: str,
    partial_sku_search: bool,
) -> list[dict[str, Any]]:
    """Filters the curated Fisher Scientific Certificate dataset using the same matching rules as fishersci.com."""
    cat_upper = catalog_number.strip().upper()
    lot_upper = lot_number.strip().upper() if lot_number else None
    cert_lower = certificate_type.strip().lower() if certificate_type else "all certificates"

    matched: list[dict[str, Any]] = []
    for asset in CURATED_CERTIFICATE_ASSETS:
        skus = [str(s).upper() for s in asset.get("sku", [])]
        root_skus = [str(r).upper() for r in asset.get("rootSku", [])]
        mpns = [str(m).upper() for m in asset.get("mpn", [])]
        all_codes = skus + root_skus + mpns

        if partial_sku_search:
            sku_ok = any(code.startswith(cat_upper) or cat_upper in code for code in all_codes)
        else:
            sku_ok = cat_upper in all_codes

        if not sku_ok:
            continue

        if lot_upper:
            asset_lot = str(asset.get("lotNumber", "")).upper()
            if asset_lot != lot_upper and not asset_lot.startswith(lot_upper):
                continue

        if cert_lower and cert_lower != "all certificates":
            asset_doc_type = str(asset.get("documentType", "")).lower()
            if cert_lower not in asset_doc_type:
                continue

        matched.append(asset)

    return matched


# ============================================================================
# 1. Tool: Validate Catalog Number (Mandatory, >= 2 chars) & Lot Number (Optional)
# ============================================================================


@trace_span("thermofisher.tool.validate_catalog_and_lot_parameters", component="tools")
def validate_catalog_and_lot_parameters(
    catalog_number: str,
    lot_number: str | None = None,
) -> CatalogValidationOutput:
    """Validates and normalizes a mandatory Catalog Number (minimum 2 characters) and optional Lot Number.

    Args:
        catalog_number: Mandatory ThermoFisher / Fisher Scientific Catalog Number (minimum 2 characters).
        lot_number: Optional manufacturing Lot Number.

    Returns:
        Strict `CatalogValidationOutput` Pydantic model with guided `error_recovery` if invalid.
    """
    cleaned_cat = (catalog_number or "").strip().upper()
    cleaned_lot = lot_number.strip().upper() if lot_number and lot_number.strip() else None

    if len(cleaned_cat) < 2:
        return CatalogValidationOutput(
            status="error",
            is_valid=False,
            normalized_catalog_number=cleaned_cat,
            normalized_lot_number=cleaned_lot,
            catalog_char_length=len(cleaned_cat),
            detected_brand="Unknown (Invalid Catalog Number)",
            validation_message=(
                f"Invalid Catalog Number '{catalog_number}'. On Fisher Scientific "
                f"(https://www.fishersci.com/search/documents?type=certificates), Catalog Number is MANDATORY "
                f"and must be a minimum of 2 characters."
            ),
            error_recovery=ErrorRecoveryGuidance(
                error_code="CATALOG_NUMBER_TOO_SHORT",
                error_message=(
                    f"Catalog number '{catalog_number}' has length {len(cleaned_cat)}, which violates the "
                    "mandatory minimum length of 2 characters."
                ),
                invalid_parameter="catalog_number",
                provided_value=str(catalog_number),
                allowed_values=[
                    "Minimum 2 characters (e.g., 'AC29', 'AC299131000', 'AC295670050', 'A452-4', 'BP160-500')"
                ],
                llm_recovery_instruction=(
                    "Do NOT guess a catalog number. Inform the user that ThermoFisher / Fisher Scientific requires "
                    "a Catalog Number with a minimum of 2 characters (e.g., 'AC29' or 'AC299131000') and ask them "
                    "to provide a valid Catalog Number."
                ),
                suggested_tool_call_example=(
                    "validate_catalog_and_lot_parameters(catalog_number='AC29', lot_number='A0487095')"
                ),
            ),
        )

    if not _VALID_CODE_PATTERN.match(cleaned_cat):
        return CatalogValidationOutput(
            status="error",
            is_valid=False,
            normalized_catalog_number=cleaned_cat,
            normalized_lot_number=cleaned_lot,
            catalog_char_length=len(cleaned_cat),
            detected_brand="Unknown (Malformed Catalog Number)",
            validation_message=(
                f"Catalog number '{catalog_number}' contains unsupported special characters. "
                "Only alphanumeric characters, hyphens, slashes, and periods are permitted."
            ),
            error_recovery=ErrorRecoveryGuidance(
                error_code="INVALID_CATALOG_NUMBER_FORMAT",
                error_message=f"Catalog number '{catalog_number}' contains invalid characters.",
                invalid_parameter="catalog_number",
                provided_value=str(catalog_number),
                allowed_values=["Alphanumeric SKU strings >= 2 chars (e.g., 'AC29', 'AC299131000', 'A452-4')"],
                llm_recovery_instruction=(
                    "Strip any special punctuation or query syntax from `catalog_number` and re-invoke with "
                    "a clean alphanumeric catalog code of at least 2 characters."
                ),
                suggested_tool_call_example="validate_catalog_and_lot_parameters(catalog_number='AC29')",
            ),
        )

    if cleaned_lot is not None and (len(cleaned_lot) < 2 or not _VALID_CODE_PATTERN.match(cleaned_lot)):
        return CatalogValidationOutput(
            status="error",
            is_valid=False,
            normalized_catalog_number=cleaned_cat,
            normalized_lot_number=cleaned_lot,
            catalog_char_length=len(cleaned_cat),
            detected_brand=_infer_brand_from_catalog(cleaned_cat),
            validation_message=(
                f"Optional Lot Number '{lot_number}' is malformed. When provided, Lot Number must be at least "
                "2 alphanumeric characters (e.g., 'A0487095'), or omitted (`lot_number=None`)."
            ),
            error_recovery=ErrorRecoveryGuidance(
                error_code="INVALID_LOT_NUMBER_FORMAT",
                error_message=f"Lot number '{lot_number}' is invalid.",
                invalid_parameter="lot_number",
                provided_value=str(lot_number),
                allowed_values=[
                    "None (omit lot_number to search all lots for the catalog number)",
                    "Valid alphanumeric lot code (e.g., 'A0487095', 'A0487009', 'A0435465')",
                ],
                llm_recovery_instruction=(
                    "Either omit `lot_number` (`lot_number=None`) to search by Catalog Number alone, or pass a "
                    "valid alphanumeric Lot Number."
                ),
                suggested_tool_call_example=(
                    f"search_certificates_of_analysis(catalog_number='{cleaned_cat}', lot_number=None)"
                ),
            ),
        )

    brand = _infer_brand_from_catalog(cleaned_cat)
    lot_msg = f" and optional Lot Number '{cleaned_lot}'" if cleaned_lot else " (no Lot Number filter; returning all lots)"
    return CatalogValidationOutput(
        status="success",
        is_valid=True,
        normalized_catalog_number=cleaned_cat,
        normalized_lot_number=cleaned_lot,
        catalog_char_length=len(cleaned_cat),
        detected_brand=brand,
        validation_message=f"Validated mandatory Catalog Number '{cleaned_cat}' ({len(cleaned_cat)} chars, {brand}){lot_msg}.",
        error_recovery=None,
    )


# ============================================================================
# 2. Tool: Search Certificates of Analysis on ThermoFisher / Fisher Scientific
# ============================================================================


@trace_span("thermofisher.tool.search_certificates_of_analysis", component="tools")
def search_certificates_of_analysis(
    catalog_number: str,
    lot_number: str | None = None,
    certificate_type: str = "Certificate of Analysis",
    page_no: int = 1,
    results_per_page: int = 10,
    partial_sku_search: bool = True,
    prefer_live_api: bool = True,
) -> CertificateSearchOutput:
    """Searches ThermoFisher / Fisher Scientific (`https://www.fishersci.com/search/documents?type=certificates`)
    for Certificates of Analysis based on a mandatory Catalog Number (>= 2 chars) and optional Lot Number.

    Args:
        catalog_number: Mandatory Catalog Number or SKU prefix (minimum 2 characters, e.g., 'AC29').
        lot_number: Optional manufacturing Lot Number (e.g., 'A0487095').
        certificate_type: Certificate type filter (default: 'Certificate of Analysis').
        page_no: Page number (1-indexed, default 1).
        results_per_page: Number of rows per page (default 10, max 60).
        partial_sku_search: True to match catalog prefixes like 'AC29' -> 'AC299131000, AC299130250'.
        prefer_live_api: True to query the live Fisher Scientific API first with automatic fallback.

    Returns:
        Strict `CertificateSearchOutput` Pydantic model containing the 5-column Fisher Scientific results.
    """
    validation = validate_catalog_and_lot_parameters(catalog_number=catalog_number, lot_number=lot_number)
    if validation.status == "error":
        return CertificateSearchOutput(
            status="error",
            data_source="validation_error",
            catalog_number_queried=str(catalog_number),
            lot_number_queried=lot_number,
            certificate_type_queried=certificate_type,
            total_results=0,
            page_no=max(1, page_no),
            results_per_page=max(1, results_per_page),
            results=[],
            summary=validation.validation_message,
            error_recovery=validation.error_recovery,
        )

    if certificate_type not in ALLOWED_CERTIFICATE_TYPE_NAMES:
        return CertificateSearchOutput(
            status="error",
            data_source="validation_error",
            catalog_number_queried=validation.normalized_catalog_number,
            lot_number_queried=validation.normalized_lot_number,
            certificate_type_queried=certificate_type,
            total_results=0,
            page_no=1,
            results_per_page=10,
            results=[],
            summary=f"Unsupported certificate_type '{certificate_type}'.",
            error_recovery=ErrorRecoveryGuidance(
                error_code="UNSUPPORTED_CERTIFICATE_TYPE",
                error_message=f"Certificate type '{certificate_type}' is not supported by Fisher Scientific.",
                invalid_parameter="certificate_type",
                provided_value=str(certificate_type),
                allowed_values=ALLOWED_CERTIFICATE_TYPE_NAMES,
                llm_recovery_instruction=(
                    "Re-invoke `search_certificates_of_analysis` with `certificate_type='Certificate of Analysis'` "
                    "or one of the allowed Fisher Scientific certificate types."
                ),
                suggested_tool_call_example=(
                    f"search_certificates_of_analysis(catalog_number='{validation.normalized_catalog_number}', "
                    "certificate_type='Certificate of Analysis')"
                ),
            ),
        )

    if page_no < 1 or results_per_page < 1 or results_per_page > 60:
        return CertificateSearchOutput(
            status="error",
            data_source="validation_error",
            catalog_number_queried=validation.normalized_catalog_number,
            lot_number_queried=validation.normalized_lot_number,
            certificate_type_queried=certificate_type,
            total_results=0,
            page_no=0,
            results_per_page=0,
            results=[],
            summary="Invalid pagination parameters: page_no must be >= 1 and results_per_page must be between 1 and 60.",
            error_recovery=ErrorRecoveryGuidance(
                error_code="INVALID_PAGINATION_PARAMETERS",
                error_message=f"Invalid pagination: page_no={page_no}, results_per_page={results_per_page}.",
                invalid_parameter="page_no" if page_no < 1 else "results_per_page",
                provided_value=str(page_no if page_no < 1 else results_per_page),
                allowed_values=["page_no >= 1", "1 <= results_per_page <= 60"],
                llm_recovery_instruction=(
                    "Re-invoke `search_certificates_of_analysis` with `page_no=1` and `results_per_page=10`."
                ),
                suggested_tool_call_example=(
                    f"search_certificates_of_analysis(catalog_number='{validation.normalized_catalog_number}', "
                    "page_no=1, results_per_page=10)"
                ),
            ),
        )

    norm_cat = validation.normalized_catalog_number
    norm_lot = validation.normalized_lot_number

    raw_assets: list[dict[str, Any]] | None = None
    data_source: Any = "curated_fallback_catalog"

    if prefer_live_api:
        raw_assets = _fetch_live_fishersci_certificates(
            catalog_number=norm_cat,
            lot_number=norm_lot,
            certificate_type=certificate_type,
            page_no=page_no,
            results_per_page=results_per_page,
            partial_sku_search=partial_sku_search,
        )
        if raw_assets is not None and len(raw_assets) > 0:
            data_source = "fishersci_live_api"

    if not raw_assets:
        curated = _search_curated_catalog(
            catalog_number=norm_cat,
            lot_number=norm_lot,
            certificate_type=certificate_type,
            partial_sku_search=partial_sku_search,
        )
        start_idx = (page_no - 1) * results_per_page
        raw_assets = curated[start_idx : start_idx + results_per_page]
        if data_source != "fishersci_live_api":
            data_source = "curated_fallback_catalog"

    records = [_asset_dict_to_schema(asset, norm_cat) for asset in raw_assets]

    lot_clause = f" and Lot Number `{norm_lot}`" if norm_lot else ""
    if records:
        summary = (
            f"Found {len(records)} `{certificate_type}` record(s) on ThermoFisher / Fisher Scientific "
            f"(`{FISHERSCI_CERTIFICATES_PORTAL_URL}`) for Catalog Number `{norm_cat}`{lot_clause}."
        )
    else:
        summary = (
            f"No `{certificate_type}` records found on Fisher Scientific for Catalog Number `{norm_cat}`{lot_clause}. "
            "Verify the Catalog Number and Lot Number on your product label, or submit a Customer Certificate Request Form."
        )

    structured_logger.info(
        f"Executed Certificate of Analysis search for catalog='{norm_cat}', lot='{norm_lot}' -> {len(records)} results ({data_source})",
        component="tools",
        event_type="coa_search_completed",
        attributes={
            "catalog_number": norm_cat,
            "lot_number": norm_lot,
            "certificate_type": certificate_type,
            "results_count": len(records),
            "data_source": data_source,
        },
    )

    return CertificateSearchOutput(
        status="success",
        source_url=FISHERSCI_CERTIFICATES_PORTAL_URL,
        api_endpoint=FISHERSCI_CERTIFICATES_API_URL,
        data_source=data_source,
        catalog_number_queried=norm_cat,
        lot_number_queried=norm_lot,
        certificate_type_queried=certificate_type,
        total_results=len(records),
        page_no=page_no,
        results_per_page=results_per_page,
        results=records,
        summary=summary,
        error_recovery=None,
    )


# ============================================================================
# 3. Tool: Get Certificate Lot Details
# ============================================================================


@trace_span("thermofisher.tool.get_certificate_lot_details", component="tools")
def get_certificate_lot_details(
    catalog_number: str,
    lot_number: str,
    prefer_live_api: bool = True,
) -> LotDetailAnalysisOutput:
    """Retrieves in-depth Certificate of Analysis metadata and verification guidance for a specific Catalog & Lot Number.

    Args:
        catalog_number: Mandatory Catalog Number (minimum 2 characters, e.g., 'AC29' or 'AC299131000').
        lot_number: Specific manufacturing Lot Number (e.g., 'A0487095').
        prefer_live_api: Whether to query the live Fisher Scientific API first.

    Returns:
        Strict `LotDetailAnalysisOutput` Pydantic model.
    """
    validation = validate_catalog_and_lot_parameters(catalog_number=catalog_number, lot_number=lot_number)
    if validation.status == "error":
        return LotDetailAnalysisOutput(
            status="error",
            catalog_number=str(catalog_number),
            lot_number=str(lot_number),
            found=False,
            record=None,
            brand_division="Validation Error",
            quality_compliance_notes=[],
            portal_verification_steps=[],
            error_recovery=validation.error_recovery,
        )

    if not validation.normalized_lot_number:
        return LotDetailAnalysisOutput(
            status="error",
            catalog_number=validation.normalized_catalog_number,
            lot_number=str(lot_number),
            found=False,
            record=None,
            brand_division=validation.detected_brand,
            quality_compliance_notes=[],
            portal_verification_steps=[],
            error_recovery=ErrorRecoveryGuidance(
                error_code="MISSING_REQUIRED_LOT_NUMBER",
                error_message="`get_certificate_lot_details` requires a non-empty `lot_number`.",
                invalid_parameter="lot_number",
                provided_value=str(lot_number),
                allowed_values=["Non-empty Lot Number string (e.g., 'A0487095', 'A0487009', 'A0435465')"],
                llm_recovery_instruction=(
                    "If the user did not provide a lot number, call `search_certificates_of_analysis` with "
                    "`lot_number=None` instead. Only call `get_certificate_lot_details` when inspecting a specific lot."
                ),
                suggested_tool_call_example=(
                    f"get_certificate_lot_details(catalog_number='{validation.normalized_catalog_number}', lot_number='A0487095')"
                ),
            ),
        )

    search_res = search_certificates_of_analysis(
        catalog_number=validation.normalized_catalog_number,
        lot_number=validation.normalized_lot_number,
        prefer_live_api=prefer_live_api,
    )
    matched = search_res.results[0] if search_res.results else None

    if matched is None:
        return LotDetailAnalysisOutput(
            status="success",
            catalog_number=validation.normalized_catalog_number,
            lot_number=validation.normalized_lot_number,
            found=False,
            record=None,
            brand_division=validation.detected_brand,
            quality_compliance_notes=[
                f"No Certificate of Analysis was found for Catalog `{validation.normalized_catalog_number}` and Lot `{validation.normalized_lot_number}`.",
                "Confirm that the Lot # matches the alphanumeric batch code printed on the bottle/vial label.",
            ],
            portal_verification_steps=[
                f"1. Open {FISHERSCI_CERTIFICATES_PORTAL_URL}",
                "2. Select 'Certificate of Analysis' from the certificate type dropdown.",
                f"3. Enter '{validation.normalized_catalog_number}' in the Catalog Number field and leave Lot Number blank to view all available lots.",
            ],
            error_recovery=None,
        )

    return LotDetailAnalysisOutput(
        status="success",
        catalog_number=validation.normalized_catalog_number,
        lot_number=validation.normalized_lot_number,
        found=True,
        record=matched,
        brand_division=validation.detected_brand,
        quality_compliance_notes=[
            f"Lot `{matched.lot_number}` of **{matched.product_name}** is certified under `{matched.certificate_type}` (Published: {matched.date}).",
            f"Covered Catalog Number(s): `{', '.join(matched.catalog_numbers)}` | Root SKU: `{', '.join(matched.root_sku)}` | ERP System: `{matched.erp}`.",
            f"eCertificate Asset Path: `{matched.ecertificate_path}` (Document UUID: `{matched.document_id}`).",
        ],
        portal_verification_steps=[
            f"1. Navigate to ThermoFisher / Fisher Scientific Certificates Search: {FISHERSCI_CERTIFICATES_PORTAL_URL}",
            f"2. Ensure '{matched.certificate_type}' is selected in the dropdown.",
            f"3. Enter Catalog Number `{validation.normalized_catalog_number}` and Lot Number `{matched.lot_number}`, then click **Search**.",
            f"4. Click the blue Lot # link (`{matched.lot_number}`) in the first column to download the signed PDF Certificate of Analysis.",
        ],
        error_recovery=None,
    )


# ============================================================================
# 4. Tool: Compare Multiple Certificate Lots
# ============================================================================


@trace_span("thermofisher.tool.compare_certificate_lots", component="tools")
def compare_certificate_lots(
    catalog_number: str,
    lot_numbers: list[str],
    prefer_live_api: bool = True,
) -> LotComparisonOutput:
    """Compares two or more manufacturing Lot Numbers for a mandatory Catalog Number (>= 2 characters).

    Args:
        catalog_number: Mandatory Catalog Number (minimum 2 characters, e.g., 'AC29').
        lot_numbers: List of 2 or more Lot Numbers to compare (e.g., ['A0487095', 'A0487009']).
        prefer_live_api: Whether to query the live Fisher Scientific API first.

    Returns:
        Strict `LotComparisonOutput` Pydantic model.
    """
    validation = validate_catalog_and_lot_parameters(catalog_number=catalog_number)
    if validation.status == "error":
        return LotComparisonOutput(
            status="error",
            catalog_number=str(catalog_number),
            lots_requested=lot_numbers or [],
            matched_records=[],
            missing_lots=[],
            most_recent_lot=None,
            comparison_summary=validation.validation_message,
            error_recovery=validation.error_recovery,
        )

    cleaned_lots = [lot.strip().upper() for lot in (lot_numbers or []) if lot and lot.strip()]
    if len(cleaned_lots) < 2:
        return LotComparisonOutput(
            status="error",
            catalog_number=validation.normalized_catalog_number,
            lots_requested=cleaned_lots,
            matched_records=[],
            missing_lots=[],
            most_recent_lot=None,
            comparison_summary="At least 2 Lot Numbers are required for `compare_certificate_lots`.",
            error_recovery=ErrorRecoveryGuidance(
                error_code="INSUFFICIENT_LOTS_FOR_COMPARISON",
                error_message=f"Provided {len(cleaned_lots)} lot(s); at least 2 lot numbers are required to compare.",
                invalid_parameter="lot_numbers",
                provided_value=str(lot_numbers),
                allowed_values=["List of >= 2 lot numbers (e.g., ['A0487095', 'A0487009'])"],
                llm_recovery_instruction=(
                    "Pass at least two Lot Numbers in `lot_numbers`, or use `search_certificates_of_analysis` "
                    "to list available lots first."
                ),
                suggested_tool_call_example=(
                    f"compare_certificate_lots(catalog_number='{validation.normalized_catalog_number}', "
                    "lot_numbers=['A0487095', 'A0487009'])"
                ),
            ),
        )

    all_search = search_certificates_of_analysis(
        catalog_number=validation.normalized_catalog_number,
        lot_number=None,
        results_per_page=50,
        prefer_live_api=prefer_live_api,
    )
    by_lot = {rec.lot_number.upper(): rec for rec in all_search.results}

    matched_records: list[CertificateRecordSchema] = []
    missing_lots: list[str] = []

    for lot in cleaned_lots:
        if lot in by_lot:
            matched_records.append(by_lot[lot])
        else:
            single = search_certificates_of_analysis(
                catalog_number=validation.normalized_catalog_number,
                lot_number=lot,
                prefer_live_api=prefer_live_api,
            )
            if single.results:
                matched_records.append(single.results[0])
            else:
                missing_lots.append(lot)

    most_recent: str | None = None
    if matched_records:
        sorted_recs = sorted(matched_records, key=lambda r: r.publish_date_iso, reverse=True)
        most_recent = sorted_recs[0].lot_number

    summary = (
        f"Compared {len(cleaned_lots)} lot(s) for Catalog `{validation.normalized_catalog_number}`: "
        f"{len(matched_records)} matched ({', '.join(r.lot_number for r in matched_records)}), "
        f"{len(missing_lots)} missing. Most recently published lot: `{most_recent or 'N/A'}`."
    )
    return LotComparisonOutput(
        status="success",
        catalog_number=validation.normalized_catalog_number,
        lots_requested=cleaned_lots,
        matched_records=matched_records,
        missing_lots=missing_lots,
        most_recent_lot=most_recent,
        comparison_summary=summary,
        error_recovery=None,
    )


# ============================================================================
# 5. Tool: List Supported Fisher Scientific Certificate Types
# ============================================================================


@trace_span("thermofisher.tool.list_supported_certificate_types", component="tools")
def list_supported_certificate_types() -> SupportedCertificateTypesOutput:
    """Returns all official Certificate Types supported on `https://www.fishersci.com/search/documents?type=certificates`."""
    items = [CertificateTypeInfoSchema(**entry) for entry in SUPPORTED_CERTIFICATE_TYPES]
    return SupportedCertificateTypesOutput(
        status="success",
        portal_url=FISHERSCI_CERTIFICATES_PORTAL_URL,
        default_certificate_type="Certificate of Analysis",
        supported_types=items,
        error_recovery=None,
    )


# ============================================================================
# 6. Tool: Check ThermoFisher Certificate of Analysis Domain Scope
# ============================================================================


def extract_catalog_and_lot_from_text(prompt: str) -> tuple[str | None, list[str]]:
    """Deterministically extracts Catalog Number and optional Lot Number(s) from natural-language user queries."""
    cleaned = prompt.strip()
    lots: list[str] = []

    # 1. Explicit lot number mentions: e.g. "lot number A0487095", "lot # A0487095", "lots A0487095 and A0487009"
    lot_Section_matches = re.finditer(
        r"\blot(?:s|\s*numbers?|\s*#|\s*no\.?)?\s*(?:is|are|=|:)?\s*([A-Za-z0-9\-]+(?:\s*(?:,|and|&|vs\.?)\s*[A-Za-z0-9\-]+)*)",
        cleaned,
        re.IGNORECASE,
    )
    stopwords = {
        "OPTIONAL", "NUMBER", "NUMBERS", "FOR", "AND", "OR", "WITH", "THE", "IS", "ARE", "NONE", "NULL", "BLANK",
        "PROVIDED", "BASED", "SEARCH", "CERTIFICATE", "ANALYSIS", "CATALOG",
    }
    for m in lot_Section_matches:
        raw_group = m.group(1)
        for token in re.split(r"(?:,|\band\b|&|\bvs\.?\b|\s+)", raw_group, flags=re.IGNORECASE):
            tok = token.strip(".,;:()[]{}\"'").upper()
            if tok and tok not in stopwords and any(ch.isdigit() for ch in tok):
                if tok not in lots:
                    lots.append(tok)

    # Also detect standalone Fisher/Acros lot patterns like A0487095, A0487009, A0435465, ZK394812
    for m in re.finditer(r"\b(A0\d{6}|ZK\d{6})\b", cleaned, re.IGNORECASE):
        tok = m.group(1).upper()
        if tok not in lots:
            lots.append(tok)

    # 2. Explicit catalog number mentions: e.g. "catalog number AC29", "catalog # A", "SKU AC299131000", "cat no BP160"
    cat_match = re.search(
        r"\b(?:catalog(?:\s*numbers?|\s*#|\s*no\.?)?|cat(?:\.?\s*no\.?|\s*#)?|sku)\s*(?:is|=|:)?\s*['\"`]?([A-Za-z0-9\-]{1,25})['\"`]?",
        cleaned,
        re.IGNORECASE,
    )
    if cat_match:
        candidate = cat_match.group(1).strip(".,;:()[]{}\"'").upper()
        if candidate not in {"NUMBER", "NUMBERS", "FOR", "IS", "AND", "WITH", "OPTIONAL", "MANDATORY"}:
            return candidate, lots

    # 3. Fallback: detect known ThermoFisher / Acros / Fisher catalog codes in the prompt (e.g. AC29, AC299131000, A452-4, BP160-500, PI23225)
    for m in re.finditer(
        r"\b(AC\d{2,10}|A\d{3}(?:-[A-Za-z0-9]+)?|BP\d{3,6}(?:-\d+)?|PI\d{4,7}|S\d{3,6}|T\d{3,6})\b",
        cleaned,
        re.IGNORECASE,
    ):
        tok = m.group(1).upper()
        if tok not in lots:
            return tok, lots

    return None, lots


@trace_span("thermofisher.tool.check_thermofisher_coa_scope", component="tools")
def check_thermofisher_coa_scope(query: str) -> DomainScopeOutput:
    """Verifies whether a user query is within the ThermoFisher Scientific Certificate of Analysis specialist scope."""
    lowered = query.lower()
    catalog_cand, lot_cands = extract_catalog_and_lot_from_text(query)

    detected_out: list[str] = []
    for kw, label in OUT_OF_SCOPE_TOPICS.items():
        if re.search(rf"\b{re.escape(kw)}\b", lowered):
            detected_out.append(label)

    has_tf_signal = any(
        w in lowered
        for w in (
            "thermofisher",
            "thermo fisher",
            "fisher scientific",
            "fishersci",
            "certificate",
            "coa",
            "catalog",
            "sku",
            "lot",
            "acros",
            "methanol",
            "reagent",
            "chemical",
            "lims",
        )
    ) or bool(catalog_cand) or bool(lot_cands)

    is_in_scope = len(detected_out) == 0 or has_tf_signal

    return DomainScopeOutput(
        status="success",
        is_in_scope=is_in_scope,
        detected_out_of_scope_topics=detected_out,
        extracted_catalog_candidates=[catalog_cand] if catalog_cand else [],
        extracted_lot_candidates=lot_cands,
        policy_reminder=(
            "You are the ThermoFisher Scientific Certificate of Analysis (CoA) Specialist Agent. "
            "Decline unrelated non-laboratory requests and assist exclusively with searching, verifying, "
            "and comparing Certificates of Analysis on https://www.fishersci.com/search/documents?type=certificates."
        ),
        error_recovery=None,
    )


# ============================================================================
# 7. Tools: Human-in-the-Loop (HITL) Gated Regulatory Actions
# ============================================================================


@require_human_confirmation("HIGH_STAKES_REGULATORY_ACTION")
@trace_span("thermofisher.tool.request_regulatory_action_confirmation", component="tools")
def request_regulatory_action_confirmation(
    action_type: str,
    catalog_number: str,
    action_description: str,
    lot_number: str | None = None,
    gxp_regulated: bool = True,
) -> HighStakesRegulatoryOutput:
    """Stages a high-stakes GxP/LIMS quality release, bulk CoA export, or missing-certificate Smartsheet request
    behind a mandatory Human-in-the-Loop (HITL) confirmation gate.
    """
    return hitl_manager.request_confirmation(
        action_type=action_type,
        catalog_number=catalog_number,
        action_description=action_description,
        lot_number=lot_number,
        gxp_regulated=gxp_regulated,
    )


@trace_span("thermofisher.tool.resolve_regulatory_action_confirmation", component="tools")
def resolve_regulatory_action_confirmation(
    confirmation_id: str,
    approved: bool,
    reviewer_id: str = "qa_specialist",
    reviewer_note: str = "",
) -> ConfirmationDecisionOutput:
    """Approves or rejects a pending Human-in-the-Loop regulatory action."""
    return hitl_manager.resolve_confirmation(
        confirmation_id=confirmation_id,
        approved=approved,
        reviewer_id=reviewer_id,
        reviewer_note=reviewer_note,
    )


ALL_THERMOFISHER_TOOLS = [
    validate_catalog_and_lot_parameters,
    search_certificates_of_analysis,
    get_certificate_lot_details,
    compare_certificate_lots,
    list_supported_certificate_types,
    check_thermofisher_coa_scope,
    request_regulatory_action_confirmation,
    resolve_regulatory_action_confirmation,
]

__all__ = [
    "ALL_THERMOFISHER_TOOLS",
    "ALLOWED_CERTIFICATE_TYPE_NAMES",
    "check_thermofisher_coa_scope",
    "compare_certificate_lots",
    "extract_catalog_and_lot_from_text",
    "get_certificate_lot_details",
    "list_supported_certificate_types",
    "request_regulatory_action_confirmation",
    "resolve_regulatory_action_confirmation",
    "search_certificates_of_analysis",
    "validate_catalog_and_lot_parameters",
]
