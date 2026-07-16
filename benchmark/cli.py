"""Console entry point for TIM3 (installed as the ``tim3`` command).

Two subcommands:

``tim3 run``
    Run a registered model over one or more domains, write its predictions,
    and score them. Thin wrapper around ``models.run`` — see
    ``tim3 run --help`` for the full flag set.

``tim3 evaluate``
    Score an existing directory of prediction files (the
    ``<domain>/<task>/*.parquet`` layout produced by ``tim3 run``)
    against the local test splits, without re-running any model.

Examples:
    tim3 run --model locf \\
        --domain fusion --data_root data/fusion_data \\
        --split_dir data/fusion_data/test --output_root predictions/locf

    tim3 evaluate --predictions_root predictions/locf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from benchmark.evaluate import VALID_TASKS
from benchmark.run import run_benchmark
from benchmark.specs import (
    BenchmarkSpec,
    load_fusion_spec,
    load_healthcare_spec,
    load_tcprimed_spec,
)

logger = logging.getLogger(__name__)

# Domain name -> (spec loader, default data root).
_DOMAIN_SPECS = {
    "fusion": (load_fusion_spec, Path("data/fusion_data")),
    "healthcare": (load_healthcare_spec, Path("data/healthcare_data")),
    "tcprimed": (load_tcprimed_spec, Path("data/tcprimed_data")),
}

_USAGE = """\
usage: tim3 <command> [options]

commands:
  run        Run a registered model over one or more domains and score it.
  evaluate   Score an existing directory of prediction files.

Run `tim3 <command> --help` for the options of each command.
"""


def _print_summary(results: dict, heading: str) -> None:
    """Print the per-task / per-domain / overall score summary.

    Args:
        results: The dict returned by ``benchmark.run.run_benchmark``.
        heading: First line of the printed block (model / predictions label).
    """
    print(f"\n{heading}")
    print(f"  Overall  NMAE: {results['nmae']:.4f}  NRMSE: {results['nrmse']:.4f}")
    for domain_result in results["per_domain"]:
        print(f"\n  Domain: {domain_result['domain']}")
        for task_result in domain_result["per_task"]:
            print(
                f"    {task_result['task']:25s}  "
                f"NMAE: {task_result['nmae']:.4f}  NRMSE: {task_result['nrmse']:.4f}"
            )


def _evaluate(argv: list[str]) -> None:
    """Score an existing predictions directory against the local test splits.

    Domains are auto-discovered: a domain is evaluated when
    ``<predictions-root>/<domain>/`` exists and the domain's data root
    contains ``signals.json`` (i.e. the data has been fetched locally).

    Args:
        argv: Command-line arguments after the ``evaluate`` subcommand.
    """
    parser = argparse.ArgumentParser(
        prog="tim3 evaluate",
        description=(
            "Score an existing directory of prediction files against the local "
            "test splits, without re-running any model. Domains are "
            "auto-discovered: a domain is scored when "
            "<predictions_root>/<domain>/ exists and its data root has been "
            "fetched locally (contains signals.json)."
        ),
    )
    parser.add_argument(
        "--predictions_root",
        required=True,
        type=Path,
        help="Directory with <domain>/<task>/*.parquet prediction files.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=VALID_TASKS,
        default=None,
        help="Subset of tasks to score (default: all four).",
    )
    for domain, (_, default_root) in _DOMAIN_SPECS.items():
        parser.add_argument(
            f"--{domain}_data",
            type=Path,
            default=None,
            help=f"Override data root for the {domain} domain (default: {default_root}).",
        )
    args = parser.parse_args(argv)

    specs: dict[str, BenchmarkSpec] = {}
    split_dirs: dict[str, Path] = {}
    for domain, (loader, default_root) in _DOMAIN_SPECS.items():
        if not (args.predictions_root / domain).is_dir():
            continue
        data_root = getattr(args, f"{domain}_data") or default_root
        if not (data_root / "signals.json").exists():
            logger.warning(
                "predictions for %r present but %s/signals.json missing — "
                "skipping domain.",
                domain,
                data_root,
            )
            continue
        specs[domain] = loader(data_root)
        split_dirs[domain] = data_root / "test"

    if not specs:
        parser.error(
            f"No evaluable domains found under {args.predictions_root}. Expected "
            f"<domain>/ subdirectories matching {sorted(_DOMAIN_SPECS)} with "
            f"local data available."
        )

    try:
        results = run_benchmark(
            predictions_root=args.predictions_root,
            domain_specs=specs,
            split_dirs=split_dirs,
            tasks=args.tasks,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    _print_summary(results, f"{args.predictions_root} — {' + '.join(sorted(specs))}")


def main(argv: list[str] | None = None) -> None:
    """Dispatch to the ``run`` or ``evaluate`` subcommand.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_USAGE, end="")
        return
    command, rest = args[0], args[1:]
    if command == "run":
        # Lazy import so `tim3 evaluate` and `--help` don't load the
        # model registry (and any model dependencies) unnecessarily.
        from models.run import main as run_main

        sys.argv = ["tim3 run", *rest]
        run_main()
    elif command == "evaluate":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        _evaluate(rest)
    else:
        print(_USAGE, end="", file=sys.stderr)
        raise SystemExit(f"tim3: unknown command {command!r}")


if __name__ == "__main__":
    main()
