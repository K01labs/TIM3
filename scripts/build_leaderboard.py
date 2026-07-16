"""Regenerate LEADERBOARD.md from the submissions under leaderboard/.

Reads every ``leaderboard/<name>/`` entry, validates it, sorts by overall NMAE
(lower is better), and writes ``LEADERBOARD.md``: an overall + per-domain
summary table followed by a per-task table for each domain, every score cell
being ``NMAE / NRMSE``. Invalid entries abort the build so a malformed
submission can never reach the table.

Usage:
    uv run python scripts/build_leaderboard.py
    uv run python scripts/build_leaderboard.py --check   # fail if out of date
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_submission import validate_entry

# Display order and labels for domain columns.
_DOMAIN_COLUMNS = [
    ("fusion", "Fusion"),
    ("healthcare", "Healthcare"),
    ("tcprimed", "Climate"),
]

# Display order and labels for the per-task breakdown columns.
_TASK_COLUMNS = [
    ("forecasting", "Forecasting"),
    ("offline_imputation", "Offline imp."),
    ("causal_imputation", "Causal imp."),
    ("cross_modal", "Cross-modal"),
]


def _fmt(value: object) -> str:
    """Format a score as 3 decimals, or 'n/a' if absent."""
    return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"


def _pair(nmae: object, nrmse: object, *, bold: bool = False) -> str:
    """Format an 'NMAE / NRMSE' cell, bolding the NMAE when ``bold`` is set."""
    n = _fmt(nmae)
    if bold and n != "n/a":
        n = f"**{n}**"
    return f"{n} / {_fmt(nrmse)}"


def _summary_row(model_name: str, config: dict, results: dict) -> str:
    """Build one row of the overall + per-domain summary table (NMAE / NRMSE)."""
    # results.json stores per_domain as a list (the run_benchmark structure);
    # index it by domain name and read the per-domain scores we display.
    per_domain = {
        d["domain"]: d for d in results.get("per_domain", []) if isinstance(d, dict)
    }
    domain_cells = [
        _pair(per_domain.get(key, {}).get("nmae"), per_domain.get(key, {}).get("nrmse"))
        for key, _ in _DOMAIN_COLUMNS
    ]
    code_url = config.get("code_url", "")
    code = f"[link]({code_url})" if code_url.startswith("http") else "n/a"
    cells = [
        model_name,
        config.get("model_type", "n/a"),
        config.get("evaluation_mode", "n/a"),
        _pair(results["nmae"], results.get("nrmse"), bold=True),
        *domain_cells,
        config.get("test_data_leakage", "n/a"),
        code,
    ]
    return "| " + " | ".join(cells) + " |"


def _domain_task_scores(results: dict) -> dict[str, dict[str, dict]]:
    """Map ``{domain: {task: task_dict}}`` from a results.json dict."""
    out: dict[str, dict[str, dict]] = {}
    for d in results.get("per_domain", []):
        if not isinstance(d, dict):
            continue
        tasks = {
            t["task"]: t
            for t in d.get("per_task", [])
            if isinstance(t, dict) and "task" in t
        }
        out[d.get("domain", "")] = tasks
    return out


def render(leaderboard_root: Path) -> str:
    """Validate all entries and render the full LEADERBOARD.md text.

    Produces an overall + per-domain summary table, followed by a per-task
    table for each domain. Every score cell is ``NMAE / NRMSE``; all tables are
    sorted by overall NMAE.
    """
    entries = sorted(p for p in leaderboard_root.glob("*") if p.is_dir())
    # records: (overall_nmae, model_name, config, results), sorted by overall.
    records: list[tuple[float, str, dict, dict]] = []
    for entry in entries:
        errors = validate_entry(entry)
        if errors:
            raise SystemExit(
                f"Cannot build leaderboard: {entry} is invalid:\n"
                + "\n".join(f"  - {e}" for e in errors)
                + "\nRun: uv run python scripts/validate_submission.py --all"
            )
        config = json.loads((entry / "config.json").read_text())
        results = json.loads((entry / "results.json").read_text())
        model_name = config.get("model_name", entry.name)
        records.append((results["nmae"], model_name, config, results))

    records.sort(key=lambda r: r[0])

    lines: list[str] = [
        "# Leaderboard",
        "",
        "Each score cell is **NMAE / NRMSE** (lower is better). Overall NMAE is "
        "the mean over the three domains. Regenerate with "
        "`uv run python scripts/build_leaderboard.py`.",
        "",
        "> Numbers are comparable only across submissions evaluated on the same "
        "hosted test split. Until the data is hosted and the open scoring issues "
        "are resolved, the table is seeded with the paper's baselines and is not "
        "yet open to external submissions (see CONTRIBUTING.md).",
        "",
        "## Overall and per-domain",
        "",
    ]

    domain_headers = " | ".join(label for _, label in _DOMAIN_COLUMNS)
    domain_rule = " | ".join("---" for _ in _DOMAIN_COLUMNS)
    lines.append(
        f"| Model | Type | Mode | Overall | {domain_headers} | Leakage | Code |"
    )
    lines.append(f"|---|---|---|---|{domain_rule}|---|---|")
    lines += [
        _summary_row(name, config, results) for _, name, config, results in records
    ]

    # Per-task scores: one table per domain, rows = models (same overall ordering).
    lines += ["", "## Per-task (NMAE / NRMSE)", ""]
    task_headers = " | ".join(label for _, label in _TASK_COLUMNS)
    task_rule = " | ".join("---" for _ in _TASK_COLUMNS)
    for dkey, dlabel in _DOMAIN_COLUMNS:
        lines += [
            f"### {dlabel}",
            "",
            f"| Model | {task_headers} |",
            f"|---|{task_rule}|",
        ]
        for _, name, _config, results in records:
            tasks = _domain_task_scores(results).get(dkey, {})
            cells = [name] + [
                _pair(tasks.get(tkey, {}).get("nmae"), tasks.get(tkey, {}).get("nrmse"))
                for tkey, _ in _TASK_COLUMNS
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaderboard-root", type=Path, default=Path("leaderboard"))
    parser.add_argument("--output", type=Path, default=Path("LEADERBOARD.md"))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if LEADERBOARD.md is out of date instead of writing it.",
    )
    args = parser.parse_args()

    rendered = render(args.leaderboard_root)

    if args.check:
        current = args.output.read_text() if args.output.exists() else ""
        if current != rendered:
            print(
                f"{args.output} is out of date. Run: "
                "uv run python scripts/build_leaderboard.py"
            )
            return 1
        print(f"{args.output} is up to date.")
        return 0

    args.output.write_text(rendered)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
