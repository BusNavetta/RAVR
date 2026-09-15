from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent

from benchmark_faiss_fixed_rate import _load_dataset, _teacher  # noqa: E402
from benchmark_nested_additive_ravr import (  # noqa: E402
    ALLOWED_LENGTHS,
    DEFAULT_SEEDS,
    _evaluation_data,
    _load_bundle,
    _load_or_calibrate,
    _selections,
    _state_path,
    panel_results_dir,
    rerank_candidates_for,
)
from compute_backend import backend_report, configure_compute_backend  # noqa: E402
from nested_additive_ravr import codec_sha256  # noqa: E402
from ravr_depth_ablation import _roles  # noqa: E402

OUTPUT_DIR = HERE / "outputs" / "workload_transfer"
FIGURE_DIR = HERE / "figures"
METRIC = "teacher_recall_at_10"
DISPLAY = {"sop": "SOP", "gldv2": "GLDv2"}
STYLE = {
    "sop": dict(color="#111827", marker="*", markersize=7.0),
    "gldv2": dict(color="#2563eb", marker="D", markersize=5.0),
}
# Any other slug (a synthetic harness corpus, a third dataset) still needs a
# distinguishable line rather than sharing one grey with its neighbour.
FALLBACK = (
    dict(color="#228b22", marker="^", markersize=4.5),
    dict(color="#e67e22", marker="o", markersize=4.0),
    dict(color="#9333ea", marker="s", markersize=4.0),
)


def style_for(slug: str, order: int) -> dict:
    return STYLE.get(slug, FALLBACK[order % len(FALLBACK)])
TRANSFER_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def report_path(dataset: str, factory: str, split: str, cal: int) -> Path:
    return OUTPUT_DIR / (
        f"workload_transfer_{dataset}_{factory.lower()}_{split}_cal{cal}.json"
    )



def _needed_sets(data: dict, roles: dict, split: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    evaluation = _evaluation_data(data, roles, split)
    teacher_ids, _ = _teacher(
        data["gallery"], evaluation["queries"], evaluation["exclude_ids"]
    )
    return np.asarray(teacher_ids[:, :k], dtype=np.int64), evaluation["queries"]


def _panel_report(dataset: str, factory: str, seed: int, split: str, cal: int,
                  target: int) -> dict | None:
    path = (
        panel_results_dir(dataset)
        / f"{factory.lower()}_seed{seed}_{split}_u{target}_cal{cal}.json"
    )
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _panel_codec_hash(dataset: str, factory: str, seed: int, split: str,
                      cal: int, targets: tuple[int, ...]) -> str | None:
    """The codec hash the stored panel actually scored, if any report exists."""
    for target in targets:
        report = _panel_report(dataset, factory, seed, split, cal, target)
        if report is not None:
            return str(report.get("codec_sha256"))
    return None


def _panel_deltas(dataset: str, factory: str, seed: int, split: str, cal: int,
                  target: int) -> np.ndarray | None:
    """Per-query RAVR-minus-uniform recall from the stored panel report."""
    report = _panel_report(dataset, factory, seed, split, cal, target)
    if report is None:
        return None
    rows = report["rows"]
    ravr = np.asarray(rows["ravr"]["per_query"][METRIC], dtype=np.float64)
    uniform = np.asarray(rows["uniform"]["per_query"][METRIC], dtype=np.float64)
    return ravr - uniform


def measure(data_root: Path, dataset: str, factory: str, seeds: tuple[int, ...],
            targets: tuple[int, ...], split: str, cal: int, k: int,
            allow_recalibration: bool = False) -> dict:
    data = _load_dataset(dataset, data_root)
    roles = _roles(data)
    needed, queries = _needed_sets(data, roles, split, k)
    n_items = len(data["gallery"])
    needed_union = np.unique(needed)

    per_seed = []
    for seed in seeds:
        state_file = _state_path(data, factory, seed, cal, "fp32")
        if not state_file.exists() and not allow_recalibration:
            raise SystemExit(
                f"missing calibration state {state_file}.\n"
                "This audit reads the frozen state the panel wrote; it does "
                "not recalibrate by default, because that silently spends the "
                "calibration cost again.\n"
                "Either copy the states next to the data root, or pass "
                "--allow-recalibration: calibration is deterministic given the "
                "codec, gallery and calibration queries, and the codec hash is "
                "checked against the stored panel report before any work runs."
            )
        bundle = _load_bundle(data, factory, seed, "fp32")
        try:
            # Whether the state is reloaded or rebuilt, the audit is only
            # meaningful if it describes the codec the panel actually scored.
            expected = _panel_codec_hash(dataset, factory, seed, split, cal, targets)
            actual = codec_sha256(bundle)
            if expected is not None and expected != actual:
                raise SystemExit(
                    f"seed {seed}: bundle {actual[:12]} is not the codec the "
                    f"stored panel scored ({expected[:12]}). Auditing it would "
                    "describe a different index than the reported numbers."
                )
            if not state_file.exists():
                print(f"seed {seed}: calibrating (no stored state)", flush=True)
            state, _, _, _ = _load_or_calibrate(
                data, roles, bundle, factory, seed, cal, "fp32", False
            )
            nbits = int(bundle.metadata["config"]["nbits"])
        finally:
            bundle.close()

        coverage = np.asarray(state["coverage"], dtype=np.int64)
        seen = coverage > 0
        query_transfer = seen[needed].mean(axis=1)
        query_exposure = coverage[needed].mean(axis=1)
        gallery_exposure = float(coverage.mean())

        row = {
            "seed": int(seed),
            "gallery_coverage": float(seen.mean()),
            "needed_coverage": float(seen[needed_union].mean()),
            "gallery_exposure": gallery_exposure,
            "needed_exposure": float(coverage[needed_union].mean()),
            "query_transfer": query_transfer.tolist(),
            "query_exposure": query_exposure.tolist(),
            "median_exposure_needed": float(np.median(coverage[needed_union])),
            "median_exposure_seen": float(np.median(coverage[seen])) if seen.any() else 0.0,
            "targets": {},
        }

        row["transfer_lift"] = (
            row["needed_coverage"] / row["gallery_coverage"]
            if row["gallery_coverage"] > 0 else float("nan")
        )
        row["exposure_lift"] = (
            row["needed_exposure"] / gallery_exposure
            if gallery_exposure > 0 else float("nan")
        )

        for target in targets:
            selections = _selections(state, n_items, seed, target, nbits)
            lengths = np.asarray(ALLOWED_LENGTHS, dtype=np.float64)
            ravr_prefix = lengths[selections["ravr"]]
            deltas = _panel_deltas(dataset, factory, seed, split, cal, target)
            row["targets"][str(target)] = {
                "uniform_stages": int(target),
                "mean_prefix_all": float(ravr_prefix.mean()),
                "mean_prefix_needed": float(ravr_prefix[needed_union].mean()),
                "mean_prefix_needed_weighted": float(ravr_prefix[needed].mean()),
                # The decisive sign: does the allocator hand the items this
                # workload needs more stages than uniform would have, or fewer?
                "needed_prefix_advantage": float(
                    ravr_prefix[needed_union].mean() - target
                ),
                "needed_prefix_advantage_weighted": float(
                    ravr_prefix[needed].mean() - target
                ),
                "starved_needed_fraction": float(
                    (ravr_prefix[needed_union] < target).mean()
                ),
                "starved_needed_fraction_weighted": float(
                    (ravr_prefix[needed] < target).mean()
                ),
                "per_query_delta": None if deltas is None else deltas.tolist(),
            }
        per_seed.append(row)

    report = {
        "dataset": dataset,
        "display_name": data["dataset"],
        "factory": factory,
        "split": split,
        "calibration_queries": cal,
        "retrieval_k": k,
        "rerank_candidates": rerank_candidates_for(dataset),
        "queries": int(len(queries)),
        "gallery_items": int(n_items),
        "needed_items": int(len(needed_union)),
        "seeds": [int(s) for s in seeds],
        "targets": [int(t) for t in targets],
        "compute_backend": backend_report(),
        "per_seed": per_seed,
        "summary": _summarise(per_seed, targets),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = report_path(dataset, factory, split, cal)
    path.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    path.with_suffix(".md").write_text(_render(report), encoding="utf-8")
    print(_render(report), flush=True)
    print(f"report -> {path}", flush=True)
    return report


def _mean_std(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def _exposure_ratio(row: dict) -> np.ndarray:
    """Per-query exposure, in multiples of an average gallery item's exposure."""
    base = row["gallery_exposure"]
    values = np.asarray(row["query_exposure"], dtype=np.float64)
    return values / base if base > 0 else np.zeros_like(values)


def _summarise(per_seed: list[dict], targets: tuple[int, ...]) -> dict:
    pooled = np.concatenate([np.asarray(row["query_transfer"]) for row in per_seed])
    ratio = np.concatenate([_exposure_ratio(row) for row in per_seed])
    summary = {
        "gallery_coverage": _mean_std(row["gallery_coverage"] for row in per_seed),
        "needed_coverage": _mean_std(row["needed_coverage"] for row in per_seed),
        "transfer_lift": _mean_std(row["transfer_lift"] for row in per_seed),
        "gallery_exposure": _mean_std(row["gallery_exposure"] for row in per_seed),
        "needed_exposure": _mean_std(row["needed_exposure"] for row in per_seed),
        "exposure_lift": _mean_std(row["exposure_lift"] for row in per_seed),
        "query_transfer_pooled": {
            "mean": float(pooled.mean()),
            "median": float(np.median(pooled)),
            "fraction_below_half": float((pooled < 0.5).mean()),
        },
        "query_exposure_ratio_pooled": {
            "mean": float(ratio.mean()),
            "median": float(np.median(ratio)),
            "fraction_below_one": float((ratio < 1.0).mean()),
        },
        "targets": {},
    }
    for target in targets:
        key = str(target)
        cells = [row["targets"][key] for row in per_seed]
        entry = {
            name: _mean_std(cell[name] for cell in cells)
            for name in (
                "needed_prefix_advantage",
                "needed_prefix_advantage_weighted",
                "mean_prefix_needed",
                "mean_prefix_needed_weighted",
                "mean_prefix_all",
                "starved_needed_fraction",
                "starved_needed_fraction_weighted",
            )
        }
        entry.update(_outcome_link(per_seed, key))
        summary["targets"][key] = entry
    return summary


def _bin_edges(values: np.ndarray, signal: str) -> np.ndarray:
    """Fixed shares for the binary signal, sample quantiles for the ratio."""
    if signal == "transfer":
        return np.asarray(TRANSFER_BINS, dtype=np.float64)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 6)))
    return edges if len(edges) > 1 else np.asarray([values.min(), values.max() + 1.0])


def _bin_outcome(signal_values: np.ndarray, delta: np.ndarray,
                 signal: str) -> list[dict]:
    edges = _bin_edges(signal_values, signal)
    bins = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        last = index == len(edges) - 2
        mask = (signal_values >= low) & (
            signal_values <= high if last else signal_values < high
        )
        if not mask.any():
            bins.append({"low": float(low), "high": float(high), "queries": 0,
                         "mean_delta": None, "standard_error": None})
            continue
        values = delta[mask]
        bins.append({
            "low": float(low), "high": float(high), "queries": int(mask.sum()),
            "mean_delta": float(values.mean()),
            "standard_error": float(values.std(ddof=1) / np.sqrt(len(values)))
            if len(values) > 1 else 0.0,
        })
    return bins


def _outcome_link(per_seed: list[dict], key: str) -> dict:
    """Join both per-query transfer signals with the panel's RAVR-uniform delta.

    Both are reported because either can be the uninformative one: the binary
    share collapses when calibration covered nearly the whole gallery, and the
    exposure ratio is the weaker description when it did not.
    """
    signals: dict[str, list] = {"transfer": [], "exposure_ratio": []}
    delta = []
    for row in per_seed:
        cell = row["targets"][key]
        if cell["per_query_delta"] is None:
            continue
        signals["transfer"].append(np.asarray(row["query_transfer"], dtype=np.float64))
        signals["exposure_ratio"].append(_exposure_ratio(row))
        delta.append(np.asarray(cell["per_query_delta"], dtype=np.float64))
    if not delta:
        return {"outcome": None}
    delta = np.concatenate(delta)
    from scipy.stats import spearmanr

    outcome = {"observations": int(len(delta)), "signals": {}}
    for name, parts in signals.items():
        values = np.concatenate(parts)
        if np.ptp(values) == 0.0:
            # A constant signal cannot rank anything; say so instead of
            # reporting a meaningless correlation.
            outcome["signals"][name] = {
                "spearman_rho": None, "spearman_p": None, "constant": True,
                "bins": _bin_outcome(values, delta, name),
            }
            continue
        rho, pvalue = spearmanr(values, delta)
        outcome["signals"][name] = {
            "spearman_rho": float(rho), "spearman_p": float(pvalue),
            "constant": False, "bins": _bin_outcome(values, delta, name),
        }
    return {"outcome": outcome}


def _render(report: dict) -> str:
    summary = report["summary"]
    lines = [
        f"# Workload transfer audit -- {report['display_name']}", "",
        f"Frozen {report['factory']}, {len(report['seeds'])} seeds, "
        f"{report['queries']} `{report['split']}` queries, "
        f"{report['calibration_queries']} calibration queries, "
        f"{report['rerank_candidates']} rerank candidates.",
        "",
        "The audit reads the calibration state and the stored panel reports. It "
        "does not recalibrate, reallocate or re-evaluate anything.",
        "",
        f"The {report['queries']} evaluated queries need "
        f"{report['needed_items']} distinct gallery items "
        f"({report['needed_items'] / report['gallery_items']:.4%} of the gallery).",
        "",
        "| Quantity | Mean | Std over seeds |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("Calibration coverage of the whole gallery", "gallery_coverage"),
        ("Calibration coverage of the needed items", "needed_coverage"),
        ("Transfer lift (needed / gallery)", "transfer_lift"),
        ("Mean exposure of a gallery item", "gallery_exposure"),
        ("Mean exposure of a needed item", "needed_exposure"),
        ("Exposure lift (needed / gallery)", "exposure_lift"),
    ):
        lines.append(
            f"| {label} | {summary[key]['mean']:.4f} | {summary[key]['std']:.4f} |"
        )
    pooled = summary["query_transfer_pooled"]
    ratio = summary["query_exposure_ratio_pooled"]
    lines += [
        "",
        f"Pooled over seeds and queries, a query finds "
        f"{pooled['mean']:.4f} of its teacher top-{report['retrieval_k']} in the "
        f"calibrated set (median {pooled['median']:.4f}); "
        f"{pooled['fraction_below_half']:.4f} of queries are below one half.",
        "",
        "Coverage saturates once the candidate depth is large relative to the "
        "gallery, so the exposure ratio carries the discrimination: a query's "
        "answers are encountered "
        f"{ratio['median']:.3f} times as often as an average gallery item "
        f"(median), and {ratio['fraction_below_one']:.4f} of queries sit below "
        "the gallery average.",
        "",
        "| Target | Advantage over uniform | Weighted by demand | "
        "Needed below uniform | Weighted | RAVR stages, all items |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for target in report["targets"]:
        cell = summary["targets"][str(target)]
        lines.append(
            f"| {target} | {cell['needed_prefix_advantage']['mean']:+.3f} | "
            f"{cell['needed_prefix_advantage_weighted']['mean']:+.3f} | "
            f"{cell['starved_needed_fraction']['mean']:.4f} | "
            f"{cell['starved_needed_fraction_weighted']['mean']:.4f} | "
            f"{cell['mean_prefix_all']['mean']:.3f} |"
        )
    lines += [
        "",
        "A positive advantage means the allocator gives the items this workload "
        "needs more stages than uniform would have. A negative one means it "
        "starves them to pay for items the calibration happened to see.",
        "",
        "The unweighted column counts each distinct needed item once; the "
        "weighted one counts it once per test query that needs it. When they "
        "disagree, the weighted column is the faithful description of where the "
        "bytes went, and the gap between them says the allocator treats "
        "heavily-demanded items differently from rarely-demanded ones.",
        "",
    ]
    outcomes = [
        (target, summary["targets"][str(target)]["outcome"])
        for target in report["targets"]
        if summary["targets"][str(target)].get("outcome")
    ]
    if outcomes:
        lines += [
            "## Does transfer predict the per-query outcome",
            "",
            "| Target | Signal | Spearman rho | p | Queries |",
            "|---:|---|---:|---:|---:|",
        ]
        for target, outcome in outcomes:
            for name, cell in outcome["signals"].items():
                rho = "constant" if cell["constant"] else f"{cell['spearman_rho']:+.4f}"
                pvalue = "n/a" if cell["constant"] else f"{cell['spearman_p']:.3g}"
                lines.append(
                    f"| {target} | {name} | {rho} | {pvalue} | "
                    f"{outcome['observations']} |"
                )
        lines.append("")
    else:
        lines += [
            "No panel report was found for these seeds and targets, so the "
            "per-query outcome join is absent.", "",
        ]
    return "\n".join(lines) + "\n"



def compare(reports: dict[str, dict], stem: Path, target: int,
            signal: str = "exposure_ratio") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    def save(figure, suffix: str) -> None:
        figure.tight_layout(pad=0.35)
        for extension in (".pdf", ".png"):
            figure.savefig(
                stem.with_name(stem.name + suffix).with_suffix(extension),
                dpi=260, bbox_inches="tight", pad_inches=0.02,
            )
        plt.close(figure)

    # (a) how much of what a query needs was ever calibrated, and how often
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))
    for order, (slug, report) in enumerate(reports.items()):
        colour = style_for(slug, order)["color"]
        label = DISPLAY.get(slug, slug)
        for axis, values in (
            (axes[0], np.concatenate(
                [np.asarray(row["query_transfer"]) for row in report["per_seed"]]
            )),
            (axes[1], np.concatenate(
                [_exposure_ratio(row) for row in report["per_seed"]]
            )),
        ):
            ordered = np.sort(values)
            axis.plot(
                ordered, np.arange(1, len(ordered) + 1) / len(ordered),
                linestyle="-", linewidth=1.2, label=label, color=colour,
            )
    axes[0].set_xlabel("Share of top-10 ever calibrated", labelpad=2)
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_title("(a) Was it seen at all", pad=3.0, weight="semibold")
    axes[0].set_ylabel("Fraction of queries", labelpad=2)
    axes[1].set_xlabel("Top-10 exposure / gallery mean", labelpad=2)
    axes[1].set_xscale("log")
    axes[1].axvline(1.0, color="#9ca3af", linewidth=0.8, linestyle=":")
    axes[1].set_title("(b) How often", pad=3.0, weight="semibold")
    for axis in axes:
        _decorate(axis)
    axes[0].legend(frameon=False, loc="upper left")
    figure.subplots_adjust(wspace=0.30)
    save(figure, "_coverage")

    # (b) does the allocator spend on the items the workload needs
    figure, axis = plt.subplots(figsize=(3.6, 2.55))
    for order, (slug, report) in enumerate(reports.items()):
        targets = report["targets"]
        cells = [report["summary"]["targets"][str(t)] for t in targets]
        style = style_for(slug, order)
        label = DISPLAY.get(slug, slug)
        axis.errorbar(
            targets,
            [cell["needed_prefix_advantage"]["mean"] for cell in cells],
            yerr=[cell["needed_prefix_advantage"]["std"] for cell in cells],
            linestyle="-", linewidth=1.1, capsize=2.2, capthick=0.7,
            elinewidth=0.7, label=label, **style,
        )
        # Same quantity weighted by how many test queries need each item.  A
        # gap between the two lines is itself the finding, so both are drawn.
        axis.errorbar(
            targets,
            [cell["needed_prefix_advantage_weighted"]["mean"] for cell in cells],
            yerr=[
                cell["needed_prefix_advantage_weighted"]["std"] for cell in cells
            ],
            linestyle="--", linewidth=1.0, capsize=2.0, capthick=0.6,
            elinewidth=0.6, markerfacecolor="white",
            label=f"{label}, by demand", **style,
        )
    axis.axhline(0.0, color="#9ca3af", linewidth=0.8, linestyle=":")
    axis.set_xlabel("Uniform-equivalent prefix (stages)", labelpad=2)
    axis.set_ylabel("Stages above uniform", labelpad=2)
    axis.set_title("(c) Where the bytes go", pad=3.0, weight="semibold")
    _decorate(axis)
    axis.legend(frameon=False)
    save(figure, "_allocation")

    # (d) transfer against the outcome the panel actually recorded
    drawn = False
    figure, axis = plt.subplots(figsize=(3.6, 2.55))
    for order, (slug, report) in enumerate(reports.items()):
        outcome = report["summary"]["targets"].get(str(target), {}).get("outcome")
        if not outcome or signal not in outcome["signals"]:
            continue
        bins = [
            b for b in outcome["signals"][signal]["bins"]
            if b["mean_delta"] is not None
        ]
        if not bins:
            continue
        drawn = True
        axis.errorbar(
            [0.5 * (b["low"] + b["high"]) for b in bins],
            [b["mean_delta"] for b in bins],
            yerr=[b["standard_error"] for b in bins],
            linestyle="-", linewidth=1.1, capsize=2.2, capthick=0.7,
            elinewidth=0.7, label=DISPLAY.get(slug, slug),
            **style_for(slug, order),
        )
    if drawn:
        axis.axhline(0.0, color="#9ca3af", linewidth=0.8, linestyle=":")
        axis.set_xlabel(
            "Share of top-10 calibrated" if signal == "transfer"
            else "Top-10 exposure / gallery mean", labelpad=2,
        )
        axis.set_ylabel("RAVR minus uniform, R@10", labelpad=2)
        axis.set_title(f"(d) Outcome at {target} stages", pad=3.0, weight="semibold")
        _decorate(axis)
        axis.legend(frameon=False)
        save(figure, "_outcome")
    else:
        plt.close(figure)
        print("no per-query panel deltas available; skipped figure (c)", flush=True)

    summary = _render_comparison(reports, target, signal)
    stem.with_suffix(".md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print(f"figures -> {stem.with_name(stem.name + '_coverage').with_suffix('.png')}",
          flush=True)


def _decorate(axis) -> None:
    axis.grid(True, color="#d1d5db", linewidth=0.45, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def _render_comparison(reports: dict[str, dict], target: int, signal: str) -> str:
    lines = [
        "# Workload transfer, corpus against corpus", "",
        "Identical measurement on both corpora: the same code path, the same "
        "seeds, targets, split and calibration size. Only the corpus and its "
        "query protocol differ.", "",
        "| Corpus | Queries | Gallery coverage | Needed coverage | Transfer lift "
        "| Exposure lift | Median exposure ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for slug, report in reports.items():
        summary = report["summary"]
        lines.append(
            f"| {DISPLAY.get(slug, slug)} | {report['queries']} | "
            f"{summary['gallery_coverage']['mean']:.4f} | "
            f"{summary['needed_coverage']['mean']:.4f} | "
            f"{summary['transfer_lift']['mean']:.3f} | "
            f"{summary['exposure_lift']['mean']:.3f} | "
            f"{summary['query_exposure_ratio_pooled']['median']:.3f} |"
        )
    lines += [
        "",
        "| Corpus | Target | Advantage over uniform | Weighted by demand "
        "| Needed items below uniform | Spearman rho (outcome signal) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for slug, report in reports.items():
        for value in report["targets"]:
            cell = report["summary"]["targets"][str(value)]
            outcome = cell.get("outcome")
            entry = outcome["signals"].get(signal) if outcome else None
            if entry is None:
                rho = "n/a"
            elif entry["constant"]:
                rho = "constant"
            else:
                rho = f"{entry['spearman_rho']:+.4f}"
            lines.append(
                f"| {DISPLAY.get(slug, slug)} | {value} | "
                f"{cell['needed_prefix_advantage']['mean']:+.3f} | "
                f"{cell['needed_prefix_advantage_weighted']['mean']:+.3f} | "
                f"{cell['starved_needed_fraction']['mean']:.4f} | {rho} |"
            )
    lines += [
        "",
        "A transfer lift near 1.0 means calibration merely covered a lot of "
        "the gallery rather than the part this workload needs; the exposure "
        "lift keeps discriminating once coverage saturates.",
        "",
        f"Figure (d) is drawn at {target} stages on the `{signal}` signal.",
        "",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("measure", help="measure one corpus")
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--dataset", default="gldv2")
    run.add_argument("--factory", choices=("LSQ16x4", "RQ16x4"), default="LSQ16x4")
    run.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    run.add_argument("--targets", nargs="+", type=int, default=[10, 12, 14])
    run.add_argument("--split", choices=("validation", "test"), default="test")
    run.add_argument("--calibration-queries", type=int, default=5000)
    run.add_argument("--retrieval-k", type=int, default=10)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument(
        "--allow-recalibration", action="store_true",
        help="Rebuild a missing calibration state instead of refusing. Only "
             "needed when the states were not copied alongside the data root.",
    )

    side = sub.add_parser("compare", help="render the corpus-against-corpus figures")
    side.add_argument("--datasets", nargs="+", default=["sop", "gldv2"])
    side.add_argument("--factory", choices=("LSQ16x4", "RQ16x4"), default="LSQ16x4")
    side.add_argument("--split", choices=("validation", "test"), default="test")
    side.add_argument("--calibration-queries", type=int, default=5000)
    side.add_argument("--figure-target", type=int, default=12)
    side.add_argument(
        "--outcome-signal", choices=("exposure_ratio", "transfer"),
        default="exposure_ratio",
        help="Which per-query signal figure (d) plots against the outcome.",
    )

    args = parser.parse_args()

    if args.command == "measure":
        resolved = configure_compute_backend(args.device)
        print(f"compute backend: {resolved} ({backend_report()})", flush=True)
        measure(
            Path(args.data_root).expanduser().resolve(), args.dataset,
            args.factory, tuple(args.seeds), tuple(args.targets), args.split,
            args.calibration_queries, args.retrieval_k,
            args.allow_recalibration,
        )
        return

    reports = {}
    for dataset in args.datasets:
        path = report_path(
            dataset, args.factory, args.split, args.calibration_queries
        )
        if not path.exists():
            raise SystemExit(
                f"missing measurement for {dataset}: {path}\n"
                f"Run: workload_transfer.py measure --dataset {dataset} ..."
            )
        reports[dataset] = json.loads(path.read_text(encoding="utf-8"))
    compare(reports, FIGURE_DIR / "workload_transfer", args.figure_target,
            args.outcome_signal)


if __name__ == "__main__":
    sys.exit(main())
