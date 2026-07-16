"""Turn a directory of predictions into a leaderboard submission.

Runs the full benchmark evaluation across every domain whose data is available,
then writes a leaderboard entry under ``<output_root>/<name>/``:

    results.json   the full benchmark scores — exactly what
                   ``benchmark.run.run_benchmark`` returns (overall, per-domain,
                   per-task, and per-sample / per-signal-channel detail and skip
                   counts)
    config.json    submission metadata stub (edit before opening a PR)

The leaderboard table and the validator read only the top-level / per-domain /
per-task scores from ``results.json``; the per-sample detail is retained for
diagnostics. See ``benchmark.run.run_benchmark`` for the full structure.

Usage:
    uv run python scripts/make_submission.py \\
        --predictions-root predictions/locf \\
        --name locf
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

from benchmark.run import run_benchmark
from benchmark.specs import (
    BenchmarkSpec,
    load_fusion_spec,
    load_healthcare_spec,
    load_tcprimed_spec,
)

logger = logging.getLogger(__name__)

# Domain name -> (spec loader, default data root).
_DOMAIN_LOADERS = {
    "fusion": (load_fusion_spec, Path("data/fusion_data")),
    "healthcare": (load_healthcare_spec, Path("data/healthcare_data")),
    "tcprimed": (load_tcprimed_spec, Path("data/tcprimed_data")),
}


def _current_commit() -> str:
    """Return the short git SHA of the benchmark checkout, or 'unknown'."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).resolve().parent.parent,
            )
            .decode()
            .strip()
        )
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _config_stub(name: str) -> dict:
    """Build a config.json stub with placeholders the contributor must fill in."""
    return {
        "model_name": name,
        "model_type": "TODO: statistical | pretrained | fine-tuned | from-scratch",
        "evaluation_mode": "TODO: zero-shot | fine-tuned | trained-from-scratch",
        "checkpoint": "TODO: model id / checkpoint, or null",
        "organization": "TODO: your name or organization",
        "code_url": "TODO: link to the adapter / repo / PR used",
        "test_data_leakage": "TODO: REQUIRED — disclose any overlap between the "
        "model's training corpus and the benchmark test data (MAST, MIMIC-IV, "
        "TC PRIMED), or 'none known'",
        "preprocessing": "TODO: gridding / interpolation / context length, etc.",
        "hardware": "TODO: device used",
        "runtime_total_seconds": 0,
        "benchmark_commit": _current_commit(),
    }


def _discover_domains(
    predictions_root: Path,
    data_root_overrides: dict[str, Path],
) -> dict[str, BenchmarkSpec]:
    """Load specs for domains that have both predictions and local data.

    A domain is included only if ``predictions_root/<domain>/`` exists and the
    domain's data root contains ``signals.json`` and ``train_stats.parquet``.
    """
    specs: dict[str, BenchmarkSpec] = {}
    for domain, (loader, default_root) in _DOMAIN_LOADERS.items():
        if not (predictions_root / domain).is_dir():
            continue
        data_root = data_root_overrides.get(domain, default_root)
        if not (data_root / "signals.json").exists():
            logger.warning(
                "make_submission: predictions for %r present but %s/signals.json "
                "missing — skipping domain.",
                domain,
                data_root,
            )
            continue
        specs[domain] = loader(data_root)
    return specs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-root",
        required=True,
        type=Path,
        help="Directory with <domain>/<task>/*.parquet prediction files.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Submission / model name (becomes the leaderboard folder name).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("leaderboard"),
        help="Where to write <name>/results.json and config.json (default: leaderboard).",
    )
    for domain, (_, default_root) in _DOMAIN_LOADERS.items():
        parser.add_argument(
            f"--{domain}-data",
            type=Path,
            default=None,
            help=f"Override data root for the {domain} domain (default: {default_root}).",
        )
    args = parser.parse_args()

    overrides = {
        domain: getattr(args, f"{domain}_data")
        for domain in _DOMAIN_LOADERS
        if getattr(args, f"{domain}_data") is not None
    }

    specs = _discover_domains(args.predictions_root, overrides)
    if not specs:
        parser.error(
            f"No evaluable domains found under {args.predictions_root}. Expected "
            f"<domain>/ subdirectories matching {sorted(_DOMAIN_LOADERS)} with "
            f"local data available."
        )

    split_dirs = {
        domain: overrides.get(domain, _DOMAIN_LOADERS[domain][1]) / "test"
        for domain in specs
    }

    logger.info("Evaluating domains: %s", sorted(specs))
    full = run_benchmark(
        predictions_root=args.predictions_root,
        domain_specs=specs,
        split_dirs=split_dirs,
    )

    # Surface coverage so incomplete submissions are obvious.
    for domain in full["per_domain"]:
        for task in domain.get("per_task", []):
            if task.get("n_skipped", 0):
                logger.warning(
                    "Coverage: %s/%s skipped %d sample(s) with no prediction file "
                    "— an incomplete submission inflates the score and will be "
                    "rejected by validate_submission.py.",
                    domain["domain"],
                    task["task"],
                    task["n_skipped"],
                )

    out_dir = args.output_root / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    # results.json is the full run_benchmark output, per-sample detail included.
    (out_dir / "results.json").write_text(json.dumps(full, indent=2) + "\n")

    config_path = out_dir / "config.json"
    if config_path.exists():
        logger.info("Keeping existing %s", config_path)
    else:
        config_path.write_text(json.dumps(_config_stub(args.name), indent=2) + "\n")
        logger.info("Wrote config stub %s — edit it before opening a PR.", config_path)

    logger.info(
        "Submission written to %s  (overall NMAE %.4f, NRMSE %.4f)",
        out_dir,
        full["nmae"],
        full["nrmse"],
    )


if __name__ == "__main__":
    main()
