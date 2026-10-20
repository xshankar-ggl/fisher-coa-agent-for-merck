"""Interactive CLI for the ThermoFisher Scientific Certificate of Analysis (CoA) Specialist Agent."""

from __future__ import annotations

import argparse
import json
import sys

from thermofisher_agent.agent import run_thermofisher_query
from thermofisher_agent.tools import (
    compare_certificate_lots,
    get_certificate_lot_details,
    list_supported_certificate_types,
    search_certificates_of_analysis,
)

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    _CONSOLE = Console()
    HAS_RICH = True
except ImportError:
    _CONSOLE = None
    HAS_RICH = False


def _print_markdown(text: str, title: str | None = None) -> None:
    if HAS_RICH and _CONSOLE is not None:
        if title:
            _CONSOLE.print(Panel(Markdown(text), title=title, border_style="bright_blue"))
        else:
            _CONSOLE.print(Markdown(text))
    else:
        if title:
            print(f"\n=== {title} ===")
        print(text)


def _print_json(data: object, title: str) -> None:
    serializable = data.model_dump() if hasattr(data, "model_dump") else data
    formatted = json.dumps(serializable, indent=2, ensure_ascii=False)
    if HAS_RICH and _CONSOLE is not None:
        _CONSOLE.print(Panel(formatted, title=title, border_style="cyan"))
    else:
        print(f"\n=== {title} ===\n{formatted}")


def run_interactive_repl() -> None:
    """Starts an interactive terminal chat session with the ThermoFisher CoA Specialist Agent."""
    welcome = (
        "**Welcome to the ThermoFisher Scientific Certificate of Analysis (CoA) Specialist Agent!**\n\n"
        "I connect to `https://www.fishersci.com/search/documents?type=certificates` to search, verify, "
        "and compare **Certificates of Analysis**.\n\n"
        "- **Catalog Number**: *Mandatory (minimum of 2 characters, e.g., `AC29`, `AC299131000`, `A452-4`)*\n"
        "- **Lot Number**: *Optional (e.g., `A0487095`, `A0487009`)*\n\n"
        "*Type `exit` or `quit` to leave.*"
    )
    _print_markdown(welcome, title="🧪 ThermoFisher Scientific CoA Specialist Agent")

    while True:
        try:
            user_input = input("\n🔬 You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended. Goodbye! 🧪")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Session ended. Goodbye! 🧪")
            break

        response = run_thermofisher_query(user_input)
        _print_markdown(response, title="🧪 ThermoFisher CoA Agent")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thermofisher-coa",
        description="ThermoFisher Scientific Certificate of Analysis (CoA) Specialist Agent CLI",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Run a single natural-language Certificate of Analysis prompt.",
    )
    parser.add_argument(
        "-c",
        "--catalog",
        type=str,
        help="Mandatory Catalog Number (minimum 2 characters, e.g., 'AC29') for direct CoA search.",
    )
    parser.add_argument(
        "-l",
        "--lot",
        type=str,
        default=None,
        help="Optional Lot Number (e.g., 'A0487095') for direct CoA search.",
    )
    parser.add_argument(
        "--compare-lots",
        nargs="+",
        metavar="LOT",
        help="Compare two or more Lot Numbers for the given --catalog (e.g. --catalog AC29 --compare-lots A0487095 A0487009).",
    )
    parser.add_argument(
        "--lot-details",
        type=str,
        metavar="LOT",
        help="Get detailed Certificate of Analysis verification info for --catalog and LOT.",
    )
    parser.add_argument(
        "--certificate-types",
        action="store_true",
        help="List all supported Certificate Types on fishersci.com.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a showcase demo searching Certificate of Analysis for Catalog AC29 (matching the screenshot).",
    )
    parser.add_argument(
        "--no-live-api",
        action="store_true",
        help="Use the curated Fisher Scientific Certificate catalog without making external HTTP calls.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    prefer_live = not args.no_live_api

    if args.certificate_types:
        result = list_supported_certificate_types()
        _print_json(result, title="📜 Supported Fisher Scientific Certificate Types")
        return 0

    if args.catalog and args.compare_lots:
        result = compare_certificate_lots(
            catalog_number=args.catalog,
            lot_numbers=args.compare_lots,
            prefer_live_api=prefer_live,
        )
        _print_json(result, title=f"⚖️ Lot Comparison — Catalog {args.catalog}")
        return 0

    if args.catalog and args.lot_details:
        result = get_certificate_lot_details(
            catalog_number=args.catalog,
            lot_number=args.lot_details,
            prefer_live_api=prefer_live,
        )
        _print_json(result, title=f"🔬 Lot Details — Catalog {args.catalog} / Lot {args.lot_details}")
        return 0

    if args.catalog is not None:
        result = search_certificates_of_analysis(
            catalog_number=args.catalog,
            lot_number=args.lot,
            prefer_live_api=prefer_live,
        )
        _print_json(result, title=f"🧪 Certificate of Analysis Search — Catalog {args.catalog}")
        return 0

    if args.demo:
        demo_prompt = (
            "Search for Certificate of Analysis on ThermoFisher Scientific for catalog number AC29 "
            "and display the results table."
        )
        _print_markdown(f"**Demo Prompt:** *{demo_prompt}*", title="🔬 User Request")
        response = run_thermofisher_query(demo_prompt, prefer_live_api=prefer_live)
        _print_markdown(response, title="🧪 ThermoFisher CoA Specialist Response")
        return 0

    if args.prompt:
        response = run_thermofisher_query(args.prompt, prefer_live_api=prefer_live)
        _print_markdown(response, title="🧪 ThermoFisher CoA Agent")
        return 0

    run_interactive_repl()
    return 0


if __name__ == "__main__":
    sys.exit(main())
