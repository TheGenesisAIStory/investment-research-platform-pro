"""Analysis Studio command line interface."""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable

from src.analysis import (
    CompetitiveAnalysisEngine,
    EarningsAnalysisEngine,
    MacroAnalysisEngine,
    PortfolioBuilderEngine,
    PortfolioRiskEngine,
    QuantResearchEngine,
    StockScreenerEngine,
    TechnicalAnalysisEngine,
)
from src.reporting import export_analysis_report


def _emit(engine, data, as_json: bool = False) -> int:
    result = engine.last_result
    manifest = export_analysis_report(result if result is not None else data, analysis_name=engine.analysis_name, summary=engine.last_summary, charts=engine.last_charts)
    if as_json:
        print(json.dumps(manifest, indent=2))
    else:
        print(engine.last_summary)
        print(f"CSV : {manifest['csv_path']}")
        print(f"HTML: {manifest['html_path']}")
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    engine = StockScreenerEngine(universe=args.universe, sector=args.sector)
    data = engine.run(top_n=args.top_n)
    return _emit(engine, data, args.json)


def cmd_risk(args: argparse.Namespace) -> int:
    engine = PortfolioRiskEngine(portfolio_file=args.portfolio_file)
    data = engine.run()
    return _emit(engine, data, args.json)


def cmd_earnings(args: argparse.Namespace) -> int:
    engine = EarningsAnalysisEngine(ticker=args.ticker)
    data = engine.run()
    return _emit(engine, data, args.json)


def cmd_build(args: argparse.Namespace) -> int:
    engine = PortfolioBuilderEngine(universe=args.universe, risk_profile=args.risk_profile)
    data = engine.run()
    return _emit(engine, data, args.json)


def cmd_technical(args: argparse.Namespace) -> int:
    engine = TechnicalAnalysisEngine(ticker=args.ticker)
    data = engine.run()
    return _emit(engine, data, args.json)


def cmd_competitive(args: argparse.Namespace) -> int:
    engine = CompetitiveAnalysisEngine(sector=args.sector)
    data = engine.run()
    return _emit(engine, data, args.json)


def cmd_quant(args: argparse.Namespace) -> int:
    engine = QuantResearchEngine(ticker=args.ticker)
    data = engine.run()
    return _emit(engine, data, args.json)


def cmd_macro(args: argparse.Namespace) -> int:
    engine = MacroAnalysisEngine(portfolio_file=args.portfolio_file)
    data = engine.run()
    return _emit(engine, data, args.json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analysis", description="Run Analysis Studio workflows.")
    parser.add_argument("--offline", action="store_true", help="Skip remote data fetches and use local/synthetic fallbacks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--json", action="store_true", help="Print report manifest as JSON.")

    screen = subparsers.add_parser("screen", help="Rank stocks by momentum, risk, and liquidity.")
    screen.add_argument("--universe", default="core")
    screen.add_argument("--sector", default=None)
    screen.add_argument("--top-n", type=int, default=10)
    add_common(screen)
    screen.set_defaults(func=cmd_screen)

    risk = subparsers.add_parser("risk", help="Analyze portfolio risk.")
    risk.add_argument("--portfolio-file", default=None)
    add_common(risk)
    risk.set_defaults(func=cmd_risk)

    earnings = subparsers.add_parser("earnings", help="Analyze earnings surprise and drift.")
    earnings.add_argument("--ticker", default="NVDA")
    add_common(earnings)
    earnings.set_defaults(func=cmd_earnings)

    build = subparsers.add_parser("build", help="Build a risk-aware portfolio.")
    build.add_argument("--risk-profile", choices=["conservative", "moderate", "aggressive"], default="moderate")
    build.add_argument("--universe", default="core")
    add_common(build)
    build.set_defaults(func=cmd_build)

    technical = subparsers.add_parser("technical", help="Run technical analysis.")
    technical.add_argument("--ticker", default="AAPL")
    add_common(technical)
    technical.set_defaults(func=cmd_technical)

    competitive = subparsers.add_parser("competitive", help="Run sector competitive analysis.")
    competitive.add_argument("--sector", default="semiconductors")
    add_common(competitive)
    competitive.set_defaults(func=cmd_competitive)

    quant = subparsers.add_parser("quant", help="Run quant factor diagnostics.")
    quant.add_argument("--ticker", default="MSFT")
    add_common(quant)
    quant.set_defaults(func=cmd_quant)

    macro = subparsers.add_parser("macro", help="Run macro portfolio analysis.")
    macro.add_argument("--portfolio-file", default=None)
    add_common(macro)
    macro.set_defaults(func=cmd_macro)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.offline:
        os.environ["ML_TRADING_OFFLINE"] = "1"
    func: Callable[[argparse.Namespace], int] = args.func
    return func(args)


if __name__ == "__main__":
    raise SystemExit(main())

