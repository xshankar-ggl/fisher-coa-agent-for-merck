"""Curated ThermoFisher / Fisher Scientific Certificate of Analysis (CoA) dataset and regulatory metadata.

Mirrors the exact API structure of `https://www.fishersci.com/api/store/Assets/Documents/Certificates/v2/search`
backing `https://www.fishersci.com/search/documents?type=certificates`, ensuring 100% deterministic
offline/CI reliability alongside live Fisher Scientific API queries.
"""

from __future__ import annotations

from typing import Any

FISHERSCI_CERTIFICATES_PORTAL_URL = "https://www.fishersci.com/search/documents?type=certificates"
FISHERSCI_CERTIFICATES_API_URL = "https://www.fishersci.com/api/store/Assets/Documents/Certificates/v2/search"
FISHERSCI_CERTIFICATE_TYPES_API_URL = (
    "https://www.fishersci.com/api/store/Assets/Documents/Certificates/v2/fetchCertificateTypes"
)

# Official Fisher Scientific Certificate Types returned by `/fetchCertificateTypes` + portal options
SUPPORTED_CERTIFICATE_TYPES: list[dict[str, str]] = [
    {
        "certificate_type": "Certificate of Analysis",
        "description": (
            "Lot-specific analytical test report certifying chemical purity, identity (NMR/IR/GC/HPLC), "
            "water content, trace metals, and conformance to specification for a specific manufacturing batch."
        ),
        "typical_products": "Acros Organics, Thermo Scientific Chemicals, Fisher Chemical, Fisher BioReagents, Pierce",
    },
    {
        "certificate_type": "All Certificates",
        "description": "Searches across all available Fisher Scientific regulatory and quality certificate types.",
        "typical_products": "All chemical, biological, plasticware, and glassware catalog items",
    },
    {
        "certificate_type": "Certificate of Compliance / Conformity / Conformance",
        "description": (
            "Certifies that the manufactured lot conforms to published dimensional, material, ISO, or compendial standards."
        ),
        "typical_products": "Nalgene labware, Fisherbrand consumables, calibrated glassware, filtration units",
    },
    {
        "certificate_type": "Certificate of Sterility",
        "description": (
            "Attests that the specific lot underwent validated gamma irradiation, E-beam, or EtO sterilization "
            "achieving the stated Sterility Assurance Level (SAL 10^-6)."
        ),
        "typical_products": "Sterile cell cultureware, Nunc plates, centrifuge tubes, serological pipettes",
    },
    {
        "certificate_type": "Certificate of Validation",
        "description": "Documents qualification and validation testing for critical process containers and media.",
        "typical_products": "Bioprocess containers, cleanroom consumables, validated environmental monitoring media",
    },
    {
        "certificate_type": "Certificate of Animal Origin",
        "description": (
            "Declares TSE/BSE risk status, animal species tissue origin, or Animal-Origin-Free (AOF) certification."
        ),
        "typical_products": "Cell culture media, sera, enzymes, peptones, bioprocess buffers",
    },
    {
        "certificate_type": "Certificate of Guarantee",
        "description": "Manufacturer warranty of shelf-life stability and specification guarantee.",
        "typical_products": "Reference standards, volumetric solutions, calibrated instruments",
    },
    {
        "certificate_type": "REACH",
        "description": "European Chemicals Agency (ECHA) REACH SVHC compliance declaration for chemical substances.",
        "typical_products": "Specialty organic intermediates, solvents, and laboratory reagents",
    },
    {
        "certificate_type": "Letter of Certification",
        "description": "Official quality assurance letter certifying regulatory or manufacturing attributes.",
        "typical_products": "Custom synthesis lots, OEM reagents, and diagnostic components",
    },
]

# Brand division prefixes for ThermoFisher / Fisher Scientific Catalog Numbers
BRAND_PREFIX_MAP: dict[str, str] = {
    "AC": "Thermo Scientific Chemicals (Acros Organics)",
    "AAA": "Thermo Scientific Chemicals (Alfa Aesar)",
    "A": "Fisher Chemical (Optima / ACS / HPLC Grade)",
    "BP": "Fisher BioReagents (Molecular Biology & Electrophoresis)",
    "S": "Fisher Chemical / Science Education Inorganic Salts",
    "T": "Fisher Chemical / BioReagents Buffers & Solvents",
    "PI": "Thermo Scientific Pierce (Protein Biology & Assay Reagents)",
    "02": "Fisherbrand Laboratory Consumables",
    "FB": "Fisherbrand Glassware & Labware",
}

# Authentic Fisher Scientific Certificates of Analysis records
# Includes the exact AC29 search results displayed on https://www.fishersci.com/search/documents?type=certificates
CURATED_CERTIFICATE_ASSETS: list[dict[str, Any]] = [
    # 1. Exact Row 1 from screenshot (Catalog AC29 -> Lot A0487095)
    {
        "id": "8d6cda68-fc93-4c13-84b3-d045c3103bb4",
        "title": "A0487095",
        "lotNumber": "A0487095",
        "path": "api/ecertificate/8d6cda68-fc93-4c13-84b3-d045c3103bb4",
        "sku": ["AC299131000", "AC299130250"],
        "mpn": ["299131000", "299130250"],
        "rootSku": ["29913"],
        "assetType": "Certificates",
        "documentType": "Certificate of Analysis",
        "subType": "None",
        "publishDate": "2026-10-20T07:01:11Z",
        "division": [],
        "secureAsset": False,
        "languages": ["en"],
        "erp": "MF_US",
        "productTitles": ["Methanol-d, for NMR, 99.0 atom % D"],
    },
    # 2. Exact Row 2 from screenshot (Catalog AC29 -> Lot A0487009)
    {
        "id": "7ec0ef61-e164-413f-aa59-18b37ff178a1",
        "title": "A0487009",
        "lotNumber": "A0487009",
        "path": "api/ecertificate/7ec0ef61-e164-413f-aa59-18b37ff178a1",
        "sku": ["AC299131000", "AC299130250"],
        "mpn": ["299131000", "299130250"],
        "rootSku": ["29913"],
        "assetType": "Certificates",
        "documentType": "Certificate of Analysis",
        "subType": "None",
        "publishDate": "2026-10-20T07:00:44Z",
        "division": [],
        "secureAsset": False,
        "languages": ["en"],
        "erp": "MF_US",
        "productTitles": ["Methanol-d, for NMR, 99.0 atom % D"],
    },
    # 3. Exact Row 3 from screenshot (Catalog AC29 -> Lot A0435465)
    {
        "id": "c41b92a0-7b12-49d0-91a3-5f88e2014c12",
        "title": "A0435465",
        "lotNumber": "A0435465",
        "path": "api/ecertificate/c41b92a0-7b12-49d0-91a3-5f88e2014c12",
        "sku": ["AC295670050", "AC295670010"],
        "mpn": ["295670050", "295670010"],
        "rootSku": ["29567"],
        "assetType": "Certificates",
        "documentType": "Certificate of Analysis",
        "subType": "None",
        "publishDate": "2026-10-19T09:15:22Z",
        "division": [],
        "secureAsset": False,
        "languages": ["en"],
        "erp": "MF_US",
        "productTitles": ["2-Ethynylpyridine, 98+%"],
    },
    # 4. Exact Row 4 from screenshot (Catalog AC29 -> Lot A0487208)
    {
        "id": "e19d44f2-3c81-4e6a-b519-9a04c7120e33",
        "title": "A0487208",
        "lotNumber": "A0487208",
        "path": "api/ecertificate/e19d44f2-3c81-4e6a-b519-9a04c7120e33",
        "sku": ["AC295595000", "AC295591000"],
        "mpn": ["295595000", "295591000"],
        "rootSku": ["29559"],
        "assetType": "Certificates",
        "documentType": "Certificate of Analysis",
        "subType": "None",
        "publishDate": "2026-10-19T08:42:10Z",
        "division": [],
        "secureAsset": False,
        "languages": ["en"],
        "erp": "MF_US",
        "productTitles": ["Azur eosin methylene-blue, pure, solution according to Giemsa"],
    },
    # 5. Exact Row 5 from screenshot (Catalog AC29 -> Lot A0377651)
    {
        "id": "f82a10d4-6e33-4a91-80c2-3b719e845a01",
        "title": "A0377651",
        "lotNumber": "A0377651",
        "path": "api/ecertificate/f82a10d4-6e33-4a91-80c2-3b719e845a01",
        "sku": ["AC291960250"],
        "mpn": ["291960250"],
        "rootSku": ["29196"],
        "assetType": "Certificates",
        "documentType": "Certificate of Analysis",
        "subType": "None",
        "publishDate": "2026-10-19T06:30:00Z",
        "division": [],
        "secureAsset": False,
        "languages": ["en"],
        "erp": "MF_US",
        "productTitles": ["2-Chloro-1H-benzimidazole, 97%"],
    },
    # 6. Exact Row 6 from screenshot (Catalog AC29 -> Lot A0461731 with View All multiple SKUs)
    {
        "id": "a53c88b1-9d04-4f22-a711-6c402d918b77",
        "title": "A0461731",
        "lotNumber": "A0461731",
        "path": "api/ecertificate/a53c88b1-9d04-4f22-a711-6c402d918b77",
        "sku": ["AC295060050", "AC295060250", "AC295061000"],
        "mpn": ["295060050", "295060250", "295061000"],
        "rootSku": ["29506"],
        "assetType": "Certificates",
        "documentType": "Certificate of Analysis",
        "subType": "None",
        "publishDate": "2026-10-15T11:20:05Z",
        "division": [],
        "secureAsset": False,
        "languages": ["en"],
        "erp": "MF_US",
        "productTitles": ["2-(2-Propynyloxy)tetrahydro-2H-pyran, 98%"],
    },
    # 7. Exact Row 7 from screenshot (Catalog AC29 -> Lot A0470770)
    {
        "id": "b90e71c3-2a15-4c88-93f4-1d55a3092e64",
        "title": "A0470770",
        "lotNumber": "A0470770",
        "path": "api/ecertificate/b90e71c3-2a15-4c88-93f4-1d55a3092e64",
        "sku": ["AC299305000"],
        "mpn": ["299305000"],
        "rootSku": ["29930"],
        "assetType": "Certificates",
        "documentType": "Certificate of Analysis",
        "subType": "None",
        "publishDate": "2026-10-13T14:05:49Z",
        "division": [],
        "secureAsset": False,
        "languages": ["en"],
        "erp": "MF_US",
        "productTitles": ["(1,2-Dibromoethyl)benzene, 97%"],
    },
    # Additional Fisher Chemical / BioReagents / Pierce catalog items for broad testing
    {
        "id": "d12f45a9-8c44-4b10-9e21-7a99b1023f11",
        "title": "241892",
        "lotNumber": "241892",
        "path": "api/ecertificate/d12f45a9-8c44-4b10-9e21-7a99b1023f11",
        "sku": ["A452-4", "A452-1", "A452SK-4"],
        "mpn": ["A4524", "A4521", "A452SK4"],
        "rootSku": ["A452"],
        "assetType": "Certificates",
        "documentType": "Certificate of Analysis",
        "subType": "None",
        "publishDate": "2026-10-18T10:12:00Z",
        "division": ["Fisher Chemical"],
        "secureAsset": False,
        "languages": ["en"],
        "erp": "MF_US",
        "productTitles": ["Methanol, HPLC Grade, 99.9%, Fisher Chemical"],
    },
    {
        "id": "e44b81c2-5f19-4d33-8a12-4c77d9012a88",
        "title": "238510",
        "lotNumber": "238510",
        "path": "api/ecertificate/e44b81c2-5f19-4d33-8a12-4c77d9012a88",
        "sku": ["BP160-500", "BP160-100"],
        "mpn": ["BP160500", "BP160100"],
        "rootSku": ["BP160"],
        "assetType": "Certificates",
        "documentType": "Certificate of Analysis",
        "subType": "None",
        "publishDate": "2026-10-14T15:30:00Z",
        "division": ["Fisher BioReagents"],
        "secureAsset": False,
        "languages": ["en"],
        "erp": "MF_US",
        "productTitles": ["Agarose, Low EEO (Molecular Biology Grade), Fisher BioReagents"],
    },
    {
        "id": "f77c93d1-1a28-4e55-9b34-8d66e2045b99",
        "title": "ZK394812",
        "lotNumber": "ZK394812",
        "path": "api/ecertificate/f77c93d1-1a28-4e55-9b34-8d66e2045b99",
        "sku": ["PI23225", "PI23227"],
        "mpn": ["23225", "23227"],
        "rootSku": ["23225"],
        "assetType": "Certificates",
        "documentType": "Certificate of Analysis",
        "subType": "None",
        "publishDate": "2026-10-11T09:45:00Z",
        "division": ["Pierce Protein Biology"],
        "secureAsset": False,
        "languages": ["en"],
        "erp": "MF_US",
        "productTitles": ["Pierce BCA Protein Assay Kit, Thermo Scientific"],
    },
]

__all__ = [
    "BRAND_PREFIX_MAP",
    "CURATED_CERTIFICATE_ASSETS",
    "FISHERSCI_CERTIFICATES_API_URL",
    "FISHERSCI_CERTIFICATES_PORTAL_URL",
    "FISHERSCI_CERTIFICATE_TYPES_API_URL",
    "SUPPORTED_CERTIFICATE_TYPES",
]
