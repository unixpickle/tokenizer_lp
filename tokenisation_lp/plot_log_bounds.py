from __future__ import annotations

import argparse
import re
from pathlib import Path


LOWER_BOUND_RE = re.compile(
    r"\bLP iteration (?P<iteration>\d+) relaxation bound: tokens>=(?P<tokens>[0-9]+(?:\.[0-9]+)?)"
)
OBJECTIVE_RE = re.compile(
    r"\bLP iteration (?P<iteration>\d+) solved\b.*\bobjective=(?P<tokens>[0-9]+(?:\.[0-9]+)?)"
)
UPPER_BOUND_RE = re.compile(
    r"\blp iteration (?P<iteration>\d+) rounded compression: .*\btokens=(?P<tokens>\d+)\b"
)


def parse_bounds(log_path: Path) -> tuple[dict[int, float], dict[int, float]]:
    lower_bounds: dict[int, float] = {}
    upper_bounds: dict[int, float] = {}

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lower_match = LOWER_BOUND_RE.search(line)
            if lower_match is not None:
                lower_bounds[int(lower_match.group("iteration"))] = float(lower_match.group("tokens"))
                continue

            objective_match = OBJECTIVE_RE.search(line)
            if objective_match is not None:
                iteration = int(objective_match.group("iteration"))
                lower_bounds.setdefault(iteration, float(objective_match.group("tokens")))
                continue

            upper_match = UPPER_BOUND_RE.search(line)
            if upper_match is not None:
                upper_bounds[int(upper_match.group("iteration"))] = float(upper_match.group("tokens"))

    return lower_bounds, upper_bounds


def plot_bounds(
    *,
    log_path: Path,
    output_path: Path,
    lower_bounds: dict[int, float],
    upper_bounds: dict[int, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    if not lower_bounds and not upper_bounds:
        raise ValueError(f"no LP bounds found in {log_path}")

    fig, axis = plt.subplots(figsize=(10, 5.5))

    if lower_bounds:
        lower_steps = sorted(lower_bounds)
        axis.plot(
            lower_steps,
            [lower_bounds[step] for step in lower_steps],
            label="LP lower bound",
            linewidth=2.0,
        )

    if upper_bounds:
        upper_steps = sorted(upper_bounds)
        axis.plot(
            upper_steps,
            [upper_bounds[step] for step in upper_steps],
            label="Rounded upper bound",
            linewidth=2.0,
        )

    axis.set_xlabel("LP cut iteration")
    axis.set_ylabel("Token count")
    axis.set_title(log_path.parent.name or log_path.name)
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LP lower bounds and rounded upper bounds from a training log.")
    parser.add_argument("log_path", type=Path, help="Path to a tokenizer-lp-train log file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path. Defaults to bounds.png next to the log.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = args.log_path.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else log_path.with_name("bounds.png")
    )
    lower_bounds, upper_bounds = parse_bounds(log_path)
    plot_bounds(
        log_path=log_path,
        output_path=output_path,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    print(
        f"wrote {output_path} "
        f"({len(lower_bounds)} lower-bound points, {len(upper_bounds)} upper-bound points)"
    )


if __name__ == "__main__":
    main()
