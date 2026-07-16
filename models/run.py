"""Unified CLI: run any registered model over one or more domains and score it.

Installed as ``tim3 run`` (see ``benchmark.cli``); also runnable as
``python -m models.run``. Pick a model with ``--model``; the harness writes
predictions for every requested task (so every model is scored on every task),
then ``run_benchmark`` prints a per-task summary. Turn the predictions into a
leaderboard entry with ``scripts/make_submission.py``.

Examples:
    # all tasks, one domain
    uv run tim3 run --model locf \\
        --domain fusion --data_root data/fusion_data \\
        --split_dir data/fusion_data/test --output_root predictions/locf

    # one task only
    uv run tim3 run --model locf --tasks forecasting \\
        --domain fusion --data_root data/fusion_data \\
        --split_dir data/fusion_data/test --output_root predictions/locf

    # several domains at once (true cross-domain score); repeat each flag in order
    uv run tim3 run --model mean \\
        --domain fusion tcprimed healthcare \\
        --data_root data/fusion_data data/tcprimed_data data/healthcare_data \\
        --split_dir data/fusion_data/test data/tcprimed_data/test data/healthcare_data/test \\
        --output_root predictions/mean
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from benchmark.evaluate import VALID_TASKS
from benchmark.run import run_benchmark
from benchmark.specs import BenchmarkSpec
from models.harness import DOMAIN_LOADERS, run_model
from models.registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    # Resolve --model first so we can attach that model's own flags. It is not
    # required here so that a bare `--help` still prints the generic options;
    # the missing-model error is raised after the full parse instead.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--model",
        choices=sorted(MODEL_REGISTRY),
        help="Registered model to run (adds that model's own flags to --help).",
    )
    known, _ = pre.parse_known_args()
    model_cls = MODEL_REGISTRY.get(known.model) if known.model else None

    parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[pre],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--domain",
        required=True,
        nargs="+",
        choices=sorted(DOMAIN_LOADERS),
        help="One or more domains to evaluate.",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        nargs="+",
        type=Path,
        help="Data directory for each domain (same order as --domain).",
    )
    parser.add_argument(
        "--split_dir",
        required=True,
        nargs="+",
        type=Path,
        help="Split directory for each domain (same order as --domain).",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        type=Path,
        help="Root directory for prediction outputs.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=VALID_TASKS,
        default=list(VALID_TASKS),
        help="Subset of tasks to run and score (default: all four).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of samples evaluated per domain (for quick checks).",
    )
    if model_cls is not None:
        model_cls.add_cli_args(parser)
    args = parser.parse_args()

    if model_cls is None:
        parser.error("the following arguments are required: --model")

    if not (len(args.domain) == len(args.data_root) == len(args.split_dir)):
        parser.error(
            "--domain, --data_root, and --split_dir must each be supplied the same "
            f"number of times (got {len(args.domain)}, {len(args.data_root)}, "
            f"{len(args.split_dir)})."
        )

    model = model_cls.from_args(args)

    domain_specs: dict[str, BenchmarkSpec] = {}
    split_dirs: dict[str, Path] = {}
    for domain_name, data_root, split_dir in zip(
        args.domain, args.data_root, args.split_dir
    ):
        spec = DOMAIN_LOADERS[domain_name](data_root)
        run_model(
            model,
            spec,
            split_dir=split_dir,
            output_root=args.output_root,
            tasks=args.tasks,
            limit=args.limit,
        )
        domain_specs[domain_name] = spec
        split_dirs[domain_name] = split_dir

    results = run_benchmark(
        predictions_root=args.output_root,
        domain_specs=domain_specs,
        split_dirs=split_dirs,
        tasks=args.tasks,
    )

    label = " + ".join(sorted(args.domain))
    print(f"\n{known.model} — {label}")
    print(f"  Overall  NMAE: {results['nmae']:.4f}  NRMSE: {results['nrmse']:.4f}")
    for domain_result in results["per_domain"]:
        print(f"\n  Domain: {domain_result['domain']}")
        for task_result in domain_result["per_task"]:
            print(
                f"    {task_result['task']:25s}  "
                f"NMAE: {task_result['nmae']:.4f}  NRMSE: {task_result['nrmse']:.4f}"
            )
    print(
        f"\nTo build a leaderboard entry from these predictions:\n"
        f"  uv run python scripts/make_submission.py "
        f"--predictions-root {args.output_root} --name {known.model}"
    )


if __name__ == "__main__":
    main()
