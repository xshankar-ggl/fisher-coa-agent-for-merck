"""Strict Pydantic v2 and TypedDict JSON schemas for the ThermoFisher Scientific Certificate of Analysis (CoA) Agent."""

from __future__ import annotations

from typing import Any, Iterator, Literal, Optional, TypedDict

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

    HAS_PYDANTIC = True
except ImportError:  # pragma: no cover - lightweight compatibility shim for bare stdlib environments
    HAS_PYDANTIC = False

    class ValidationError(ValueError):  # type: ignore[no-redef]
        """Fallback validation error when Pydantic is unavailable."""

    def ConfigDict(**kwargs: Any) -> dict[str, Any]:  # type: ignore[no-redef]
        return dict(kwargs)

    def Field(default: Any = None, *, default_factory: Any = None, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        return default_factory if default_factory is not None else default

    def field_validator(*fields: str, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        def _decorator(fn: Any) -> Any:
            return classmethod(fn) if not isinstance(fn, classmethod) else fn

        return _decorator

    class BaseModel:  # type: ignore[no-redef]
        """Minimal Pydantic BaseModel compatibility layer for stdlib-only test runners."""

        def __init__(self, **data: Any) -> None:
            annotations: dict[str, Any] = {}
            for cls in reversed(self.__class__.__mro__):
                annotations.update(getattr(cls, "__annotations__", {}))
            for key in annotations:
                if key in data:
                    setattr(self, key, data[key])
                elif hasattr(self.__class__, key):
                    attr = getattr(self.__class__, key)
                    setattr(self, key, attr() if callable(attr) else attr)
                else:
                    setattr(self, key, None)
            for key, value in data.items():
                setattr(self, key, value)

        def model_dump(self) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in self.__dict__.items():
                if key.startswith("_"):
                    continue
                if isinstance(value, BaseModel):
                    result[key] = value.model_dump()
                elif isinstance(value, list):
                    result[key] = [
                        item.model_dump() if isinstance(item, BaseModel) else item
                        for item in value
                    ]
                elif isinstance(value, dict):
                    result[key] = {
                        k: (v.model_dump() if isinstance(v, BaseModel) else v)
                        for k, v in value.items()
                    }
                else:
                    result[key] = value
            return result

        @classmethod
        def model_json_schema(cls) -> dict[str, Any]:
            return {
                "title": cls.__name__,
                "type": "object",
                "properties": {k: {"type": str(v)} for k, v in getattr(cls, "__annotations__", {}).items()},
            }


class StrictToolOutputModel(BaseModel):
    """Base Pydantic schema for all ThermoFisher CoA Agent tool outputs.

    Enforces strict JSON schema serialization (`extra="forbid"`) while supporting mapping-style
    key access (`result["field"]`) for seamless interoperability across Google ADK, FastAPI, and test suites.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the Pydantic model into a plain JSON-compatible dictionary."""
        return self.model_dump()

    def __getitem__(self, key: str) -> Any:
        dump = self.model_dump()
        if key not in dump:
            raise KeyError(key)
        return dump[key]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.model_dump()

    def get(self, key: str, default: Any = None) -> Any:
        return self.model_dump().get(key, default)

    def keys(self) -> Any:
        return self.model_dump().keys()

    def items(self) -> Any:
        return self.model_dump().items()

    def values(self) -> Any:
        return self.model_dump().values()

    def __iter__(self) -> Iterator[str]:
        return iter(self.model_dump())


# ============================================================================
# 1. Guided Error Recovery Schemas (Instructing the LLM how to self-correct)
# ============================================================================


class ErrorRecoveryGuidanceDict(TypedDict):
    """TypedDict representation of guided LLM error recovery metadata."""

    error_code: str
    error_message: str
    invalid_parameter: str
    provided_value: str
    allowed_values: list[str]
    llm_recovery_instruction: str
    suggested_tool_call_example: str


class ErrorRecoveryGuidance(StrictToolOutputModel):
    """Actionable error recovery schema that instructs the LLM on how to fix invalid tool inputs."""

    error_code: str = Field(..., description="Machine-readable error code (e.g., CATALOG_NUMBER_TOO_SHORT).")
    error_message: str = Field(..., description="Clear human- and LLM-readable explanation of the validation error.")
    invalid_parameter: str = Field(..., description="Name of the parameter that failed validation.")
    provided_value: str = Field(..., description="String representation of the invalid value passed by the caller.")
    allowed_values: list[str] = Field(
        default_factory=list,
        description="Explicit list of valid values or patterns accepted by this parameter.",
    )
    llm_recovery_instruction: str = Field(
        ...,
        description=(
            "Direct instruction to the LLM explaining how to reformulate the tool call with valid arguments "
            "or what clarifying question to ask the user."
        ),
    )
    suggested_tool_call_example: str = Field(
        ...,
        description="Concrete example of a valid invocation of this tool.",
    )


# ============================================================================
# 2. Tool Input Validation Schemas
# ============================================================================


class CertificateSearchInput(BaseModel):
    """Strict input validation schema for `search_certificates_of_analysis`."""

    model_config = ConfigDict(extra="forbid")

    catalog_number: str = Field(
        ...,
        min_length=2,
        max_length=64,
        description=(
            "Mandatory ThermoFisher / Fisher Scientific Catalog Number or SKU prefix. "
            "Must be a minimum of 2 characters (e.g., 'AC29', 'AC299131000', 'A452', 'BP160')."
        ),
    )
    lot_number: Optional[str] = Field(
        default=None,
        description="Optional manufacturing Lot Number to filter results (e.g., 'A0487095', 'A0487009').",
    )
    certificate_type: str = Field(
        default="Certificate of Analysis",
        description="Certificate type filter on Fisher Scientific (default: 'Certificate of Analysis').",
    )
    page_no: int = Field(
        default=1,
        ge=1,
        le=100,
        description="1-indexed page number for paginated Fisher Scientific results.",
    )
    results_per_page: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Number of certificate records per page (1 to 60).",
    )
    partial_sku_search: bool = Field(
        default=True,
        description="Whether prefix/partial catalog number matching is enabled (default True, matching fishersci.com).",
    )


class LotDetailsInput(BaseModel):
    """Strict input validation schema for `get_certificate_lot_details`."""

    model_config = ConfigDict(extra="forbid")

    catalog_number: str = Field(
        ...,
        min_length=2,
        description="Mandatory Catalog Number (minimum 2 characters, e.g., 'AC29' or 'AC299131000').",
    )
    lot_number: str = Field(
        ...,
        min_length=2,
        description="Target Lot Number to inspect (e.g., 'A0487095').",
    )


class LotComparisonInput(BaseModel):
    """Strict input validation schema for `compare_certificate_lots`."""

    model_config = ConfigDict(extra="forbid")

    catalog_number: str = Field(
        ...,
        min_length=2,
        description="Mandatory Catalog Number (minimum 2 characters, e.g., 'AC29').",
    )
    lot_numbers: list[str] = Field(
        ...,
        min_length=2,
        description="List of at least 2 Lot Numbers to compare (e.g., ['A0487095', 'A0487009']).",
    )


# ============================================================================
# 3. Strict Tool Output Schemas (Pydantic BaseModel + TypedDict)
# ============================================================================


class CertificateRecordDict(TypedDict):
    """TypedDict matching a single row of the Fisher Scientific Certificate Search results table."""

    document_id: str
    lot_number: str
    certificate_type: str
    date: str
    publish_date_iso: str
    catalog_numbers: list[str]
    catalog_numbers_display: str
    has_more_catalog_numbers: bool
    product_name: str
    mpn_list: list[str]
    root_sku: list[str]
    erp: str
    languages: list[str]
    ecertificate_path: str
    fishersci_search_url: str


class CertificateRecordSchema(StrictToolOutputModel):
    """Schema for a single Certificate of Analysis row returned by Fisher Scientific (`fishersci.com`).

    Maps 1-to-1 with the 5 columns displayed on `https://www.fishersci.com/search/documents?type=certificates`:
    1. `lot_number` -> **Lot #**
    2. `certificate_type` -> **Certificate Type**
    3. `date` -> **Date** (formatted e.g. `Oct 20, 2026`)
    4. `catalog_numbers` / `catalog_numbers_display` -> **Catalog Number(s)** (with `View All` indicator)
    5. `product_name` -> **Product Name**
    """

    document_id: str = Field(..., description="Unique Fisher Scientific asset UUID.")
    lot_number: str = Field(..., description="Column 1: Lot # (e.g., 'A0487095').")
    certificate_type: str = Field(..., description="Column 2: Certificate Type (e.g., 'Certificate of Analysis').")
    date: str = Field(..., description="Column 3: Date formatted as 'MMM DD, YYYY' (e.g., 'Oct 20, 2026').")
    publish_date_iso: str = Field(..., description="Raw ISO-8601 publish timestamp from Fisher Scientific API.")
    catalog_numbers: list[str] = Field(
        default_factory=list,
        description="Column 4: Full list of Catalog Number(s) / SKUs covered by this certificate.",
    )
    catalog_numbers_display: str = Field(
        ...,
        description="Formatted Catalog Number(s) display string matching Fisher Scientific UI (with '... View All' if >2 SKUs).",
    )
    has_more_catalog_numbers: bool = Field(
        default=False,
        description="True when more than 2 catalog numbers exist and the UI renders a 'View All' expander.",
    )
    product_name: str = Field(
        ...,
        description="Column 5: Product Name (e.g., 'Methanol-d, for NMR, 99.0 atom % D').",
    )
    mpn_list: list[str] = Field(default_factory=list, description="Manufacturer Part Numbers (MPNs).")
    root_sku: list[str] = Field(default_factory=list, description="Root SKU identifiers.")
    erp: str = Field(default="MF_US", description="Fisher Scientific ERP source system (e.g., 'MF_US').")
    languages: list[str] = Field(default_factory=lambda: ["en"], description="Available document languages.")
    ecertificate_path: str = Field(..., description="Fisher Scientific internal eCertificate asset path.")
    fishersci_search_url: str = Field(
        ...,
        description="Direct URL to view/verify this search on Fisher Scientific.",
    )


class CertificateSearchOutputDict(TypedDict):
    status: Literal["success", "error"]
    source_url: str
    api_endpoint: str
    data_source: Literal["fishersci_live_api", "curated_fallback_catalog", "validation_error"]
    catalog_number_queried: str
    lot_number_queried: Optional[str]
    certificate_type_queried: str
    total_results: int
    page_no: int
    results_per_page: int
    results: list[CertificateRecordDict]
    summary: str
    error_recovery: Optional[ErrorRecoveryGuidanceDict]


class CertificateSearchOutput(StrictToolOutputModel):
    """Strict output schema for `search_certificates_of_analysis`."""

    status: Literal["success", "error"] = Field(default="success", description="Tool execution status.")
    source_url: str = Field(
        default="https://www.fishersci.com/search/documents?type=certificates",
        description="ThermoFisher / Fisher Scientific Certificates Search Portal URL.",
    )
    api_endpoint: str = Field(
        default="https://www.fishersci.com/api/store/Assets/Documents/Certificates/v2/search",
        description="Underlying Fisher Scientific REST API endpoint.",
    )
    data_source: Literal["fishersci_live_api", "curated_fallback_catalog", "validation_error"] = Field(
        ...,
        description="Indicates whether results came from the live FisherSci API or the offline fallback catalog.",
    )
    catalog_number_queried: str = Field(..., description="Catalog number used for the search (mandatory, >= 2 chars).")
    lot_number_queried: Optional[str] = Field(
        default=None, description="Optional lot number used to filter the search."
    )
    certificate_type_queried: str = Field(
        default="Certificate of Analysis", description="Certificate type filter applied."
    )
    total_results: int = Field(default=0, ge=0, description="Total number of matching certificates returned.")
    page_no: int = Field(default=1, ge=0, description="Current page number.")
    results_per_page: int = Field(default=10, ge=0, description="Results per page limit.")
    results: list[CertificateRecordSchema] = Field(
        default_factory=list,
        description="Ordered list of matching Certificate of Analysis records.",
    )
    summary: str = Field(..., description="Human-readable summary of the Certificate of Analysis search results.")
    error_recovery: Optional[ErrorRecoveryGuidance] = Field(
        default=None,
        description="Populated with actionable LLM recovery instructions when status == 'error'.",
    )


class LotDetailAnalysisOutput(StrictToolOutputModel):
    """Strict output schema for `get_certificate_lot_details`."""

    status: Literal["success", "error"] = Field(default="success", description="Tool execution status.")
    catalog_number: str = Field(..., description="Queried catalog number.")
    lot_number: str = Field(..., description="Queried lot number.")
    found: bool = Field(..., description="True if a matching Certificate of Analysis record was found.")
    record: Optional[CertificateRecordSchema] = Field(
        default=None, description="Detailed certificate record if found."
    )
    brand_division: str = Field(
        default="Thermo Scientific / Acros Organics / Fisher Chemical",
        description="Identified ThermoFisher brand division based on SKU prefix.",
    )
    quality_compliance_notes: list[str] = Field(
        default_factory=list,
        description="Quality, ISO/GMP traceability, and verification guidance for this lot.",
    )
    portal_verification_steps: list[str] = Field(
        default_factory=list,
        description="Step-by-step instructions to download the official PDF on fishersci.com.",
    )
    error_recovery: Optional[ErrorRecoveryGuidance] = Field(
        default=None,
        description="Populated with LLM recovery instructions when status == 'error'.",
    )


class LotComparisonOutput(StrictToolOutputModel):
    """Strict output schema for `compare_certificate_lots`."""

    status: Literal["success", "error"] = Field(default="success", description="Tool execution status.")
    catalog_number: str = Field(..., description="Catalog number evaluated.")
    lots_requested: list[str] = Field(default_factory=list, description="Lot numbers requested for comparison.")
    matched_records: list[CertificateRecordSchema] = Field(
        default_factory=list, description="Certificate records matched for the requested lots."
    )
    missing_lots: list[str] = Field(
        default_factory=list, description="Lot numbers that did not return a certificate."
    )
    most_recent_lot: Optional[str] = Field(
        default=None, description="Lot number with the latest publication date."
    )
    comparison_summary: str = Field(..., description="Synthesized comparison of the evaluated lots.")
    error_recovery: Optional[ErrorRecoveryGuidance] = Field(
        default=None,
        description="Populated with LLM recovery instructions when status == 'error'.",
    )


class CertificateTypeInfoSchema(StrictToolOutputModel):
    """Schema describing a single Fisher Scientific certificate type."""

    certificate_type: str = Field(..., description="Official Fisher Scientific certificate type name.")
    description: str = Field(..., description="What this certificate attests to in laboratory/regulatory workflows.")
    typical_products: str = Field(..., description="Products that typically carry this certificate.")


class SupportedCertificateTypesOutput(StrictToolOutputModel):
    """Strict output schema for `list_supported_certificate_types`."""

    status: Literal["success", "error"] = Field(default="success", description="Tool execution status.")
    portal_url: str = Field(
        default="https://www.fishersci.com/search/documents?type=certificates",
        description="ThermoFisher Certificate Search Portal URL.",
    )
    default_certificate_type: str = Field(
        default="Certificate of Analysis",
        description="Default certificate type selected on the portal.",
    )
    supported_types: list[CertificateTypeInfoSchema] = Field(
        default_factory=list,
        description="List of supported certificate types on fishersci.com.",
    )
    error_recovery: Optional[ErrorRecoveryGuidance] = Field(default=None)


class CatalogValidationOutput(StrictToolOutputModel):
    """Strict output schema for `validate_catalog_and_lot_parameters`."""

    status: Literal["success", "error"] = Field(default="success", description="Validation status.")
    is_valid: bool = Field(..., description="True if catalog_number >= 2 chars and lot_number (if any) is valid.")
    normalized_catalog_number: str = Field(..., description="Trimmed, uppercase-normalized catalog number.")
    normalized_lot_number: Optional[str] = Field(default=None, description="Normalized optional lot number.")
    catalog_char_length: int = Field(..., ge=0, description="Character length of the normalized catalog number.")
    detected_brand: str = Field(..., description="ThermoFisher sub-brand inferred from catalog prefix.")
    validation_message: str = Field(..., description="Human-readable validation result.")
    error_recovery: Optional[ErrorRecoveryGuidance] = Field(
        default=None,
        description="Populated when catalog_number < 2 characters or contains invalid characters.",
    )


class DomainScopeOutput(StrictToolOutputModel):
    """Strict output schema for `check_thermofisher_coa_scope`."""

    status: Literal["success", "error"] = Field(default="success", description="Tool execution status.")
    is_in_scope: bool = Field(
        ...,
        description="True if the query pertains to ThermoFisher / Fisher Scientific certificates, catalog/lot lookups, or lab quality docs.",
    )
    detected_out_of_scope_topics: list[str] = Field(
        default_factory=list,
        description="Non-ThermoFisher or unrelated topics detected in the query.",
    )
    extracted_catalog_candidates: list[str] = Field(
        default_factory=list,
        description="Candidate catalog numbers extracted from the prompt.",
    )
    extracted_lot_candidates: list[str] = Field(
        default_factory=list,
        description="Candidate lot numbers extracted from the prompt.",
    )
    policy_reminder: str = Field(..., description="Instruction on how the agent enforces ThermoFisher CoA scope.")
    error_recovery: Optional[ErrorRecoveryGuidance] = Field(default=None)


class HighStakesRegulatoryOutput(StrictToolOutputModel):
    """Strict output schema for HITL-gated quality/regulatory actions."""

    status: Literal["pending_human_approval", "approved", "rejected", "error"] = Field(
        ..., description="Current state of the Human-in-the-Loop confirmation workflow."
    )
    requires_human_confirmation: bool = Field(
        default=True,
        description="Always True for GxP/LIMS quality releases, bulk exports, or external customer certificate requests.",
    )
    confirmation_id: str = Field(..., description="Unique tracking ID for the pending confirmation hook.")
    action_type: str = Field(..., description="Category of high-stakes regulatory/quality action.")
    catalog_number: str = Field(..., description="Associated Catalog Number (minimum 2 characters).")
    lot_number: Optional[str] = Field(default=None, description="Associated Lot Number (optional).")
    action_description: str = Field(..., description="Detailed description of the staged regulatory action.")
    regulatory_impact_warning: str = Field(
        ..., description="Disclosure of GxP / ISO / 21 CFR Part 11 regulatory impact."
    )
    approval_instructions: str = Field(
        ...,
        description="Instructions for the human QA officer or researcher to approve or reject the action.",
    )
    error_recovery: Optional[ErrorRecoveryGuidance] = Field(
        default=None,
        description="Populated with LLM recovery instructions when status == 'error'.",
    )


__all__ = [
    "BaseModel",
    "CatalogValidationOutput",
    "CertificateRecordDict",
    "CertificateRecordSchema",
    "CertificateSearchInput",
    "CertificateSearchOutput",
    "CertificateSearchOutputDict",
    "CertificateTypeInfoSchema",
    "ConfigDict",
    "DomainScopeOutput",
    "ErrorRecoveryGuidance",
    "ErrorRecoveryGuidanceDict",
    "Field",
    "HAS_PYDANTIC",
    "HighStakesRegulatoryOutput",
    "LotComparisonInput",
    "LotComparisonOutput",
    "LotDetailAnalysisOutput",
    "LotDetailsInput",
    "StrictToolOutputModel",
    "SupportedCertificateTypesOutput",
    "ValidationError",
    "field_validator",
]
