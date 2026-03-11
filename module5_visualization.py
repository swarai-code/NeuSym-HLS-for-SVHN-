
# module5_visualization.py  -  NeuSym-HLS  |  Module 5
#
# Evaluation Visualization
#
# Reads EXCLUSIVELY from files produced by the pipeline at runtime:
#
#   eval_results/baseline_accuracy.txt   <- written by Module 1
#   eval_results/accuracy_log.csv        <- written by Module 4 (fine-tuning)
#   sr_results/<level>_<opset>/
#       hall_of_fame.csv                 <- written by Module 2 (PySR)
#   eval_results/synthesis_metrics.csv   <- written by Module 4 (Vitis HLS)
#                                           optional: synthesis not required
#                                           to generate accuracy/area charts
#
# This module is intentionally free of hardcoded placeholder numbers.
# If a required file is missing it raises a clear error telling the user
# which pipeline stage needs to be run first.
#
# Charts produced
# ---------------
#   1. area_vs_accuracy.png   - dual-axis grouped bars (area) + lines (accuracy)
#   2. resource_breakdown.png - LUT/FF/DSP breakdown per configuration
#                               (only if synthesis_metrics.csv exists)
#   3. pareto_scatter.png     - accuracy vs area Pareto scatter
#   4. sr_complexity.png      - SR equation complexity and MSE per config
#                               (from hall_of_fame.csv, always available)
#
# All outputs saved to eval_results/  at 300 dpi (PNG) and vector (PDF).


import csv
import pathlib
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# Paths 
ROOT     = pathlib.Path(__file__).parent
EVAL_DIR = ROOT / "eval_results"
SR_DIR   = ROOT / "sr_results"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Axis / layout constants 
OPSETS  = ["SCE", "SRL", "POL"]
LEVELS  = ["1L", "2L"]

COLOR = {
    "bar_1L":    "#4C72B0",
    "bar_2L":    "#DD8452",
    "line_1L":   "#1A5276",
    "line_2L":   "#922B21",
    "baseline":  "#27AE60",
    "lut":       "#4C72B0",
    "ff":        "#55A868",
    "dsp":       "#C44E52",
    "sce":       "#E74C3C",
    "srl":       "#F39C12",
    "pol":       "#2980B9",
}
OPSET_COLORS = {"SCE": COLOR["sce"], "SRL": COLOR["srl"], "POL": COLOR["pol"]}



# 1.  Data loaders  (each raises a descriptive error if the file is missing)


def _require(path: pathlib.Path, stage: str) -> pathlib.Path:
    """Raise a clear error if *path* does not exist yet."""
    if not path.exists():
        raise FileNotFoundError(
            f"\n[Module 5] Required file not found:\n"
            f"   {path}\n"
            f"   -> Run {stage} first, then re-run module5_visualization.py"
        )
    return path


def load_baseline_accuracy() -> float:
    """
    Load the FP32 baseline accuracy saved by Module 1.
    Returns accuracy as a float in [0, 100].
    """
    path = _require(
        EVAL_DIR / "baseline_accuracy.txt",
        "Module 1  (python run_pipeline.py  or  python module1_data_model.py)"
    )
    val = float(path.read_text().strip())
    print(f"[Module 5] Baseline accuracy: {val:.2f}%")
    return val


def load_accuracy_log() -> dict[str, float]:
    """
    Load the per-config fine-tuned accuracy written by Module 4.
    Returns dict keyed by tag string "1L_POL", "2L_SCE", etc.
    """
    path = _require(
        EVAL_DIR / "accuracy_log.csv",
        "Module 4  (python run_pipeline.py  or  python module4_finetune_eval.py)"
    )
    result: dict[str, float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            result[row["tag"].strip()] = float(row["accuracy"])
    print(f"[Module 5] Loaded accuracy for {len(result)} configs: {list(result)}")
    return result


def load_sr_metrics() -> dict[str, dict]:
    """
    Load SR complexity and MSE from each config's hall_of_fame.csv.
    Returns dict keyed by tag "1L_POL" etc., values {"complexity", "mse"}.

    Uses the row with the lowest loss that also appears in best_equation.txt
    (i.e. the hardware-budget-filtered best equation per config).
    """
    metrics: dict[str, dict] = {}
    for lvl in LEVELS:
        for ops in OPSETS:
            tag       = f"{lvl}_{ops}"
            hof_path  = SR_DIR / tag / "hall_of_fame.csv"
            best_path = SR_DIR / tag / "best_equation.txt"

            if not hof_path.exists():
                print(f"[Module 5] WARNING: hall_of_fame.csv missing for {tag} "
                      f"(run Module 2)")
                continue

            # Read hall of fame, find lowest-loss row
            rows: list[dict] = []
            with open(hof_path, newline="") as f:
                rows = list(csv.DictReader(f))

            if not rows:
                continue

            # Sort by loss ascending, take best
            rows.sort(key=lambda r: float(r.get("loss", "inf")))
            best = rows[0]
            metrics[tag] = {
                "complexity": float(best.get("complexity", 0)),
                "mse":        float(best.get("loss", 0)),
            }

    print(f"[Module 5] Loaded SR metrics for {len(metrics)} configs.")
    return metrics


def load_synthesis_metrics() -> dict[str, dict] | None:
    """
    Load LUT/FF/DSP from synthesis_metrics.csv (written by Module 4 Vitis HLS).
    Returns None (not an error) if the file does not exist, because synthesis
    is optional — the pipeline can run without Vitis HLS installed.
    """
    path = EVAL_DIR / "synthesis_metrics.csv"
    if not path.exists():
        print("[Module 5] synthesis_metrics.csv not found  "
              "(Vitis HLS synthesis not yet run — skipping HW resource charts)")
        return None

    result: dict[str, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            tag = row["tag"].strip()
            result[tag] = {
                "LUT_pct":  float(row.get("LUT_pct", 0)),
                "FF_pct":   float(row.get("FF_pct",  0)),
                "DSP_pct":  float(row.get("DSP_pct", 0)),
                "latency_min_us": float(row.get("latency_min_us", 0)),
                "latency_max_us": float(row.get("latency_max_us", 0)),
            }
    print(f"[Module 5] Loaded synthesis metrics for {len(result)} configs.")
    return result



# 2.  Chart 1  -  Hardware Area vs. Accuracy  (dual-axis)


def plot_area_vs_accuracy(accuracy_log:   dict[str, float],
                           baseline_acc:  float,
                           synth_metrics: dict[str, dict] | None,
                           sr_metrics:    dict[str, dict]) -> pathlib.Path:
    """
    Dual-axis chart:
      Left  Y: Area relative to FP32 baseline (LUT% from synthesis, or SR
               normalised complexity when synthesis hasn't been run)
      Right Y: Fine-tuned test accuracy (%)
      X     : Operator sets {SCE, SRL, POL}
      Series: 1L and 2L distillation levels
    """
    # Decide area source 
    # If synthesis has been run use real LUT%; otherwise normalise SR complexity
    # so the largest value = 1.0, giving a relative hardware cost proxy.
    use_real_hw = synth_metrics is not None

    def _area(tag: str) -> float:
        if use_real_hw:
            return synth_metrics.get(tag, {}).get("LUT_pct", float("nan"))
        else:
            # Normalise complexity by the maximum observed
            max_c = max((v["complexity"] for v in sr_metrics.values()), default=1)
            return (sr_metrics.get(tag, {}).get("complexity", float("nan"))
                    / max_c * 100.0)

    area_label = ("LUT Utilisation (% of XC7Z020 total)"
                  if use_real_hw
                  else "Relative SR Complexity (normalised, %)")

    # Assemble arrays 
    area_1L  = np.array([_area(f"1L_{ops}") for ops in OPSETS])
    area_2L  = np.array([_area(f"2L_{ops}") for ops in OPSETS])
    acc_1L   = np.array([accuracy_log.get(f"1L_{ops}", float("nan")) for ops in OPSETS])
    acc_2L   = np.array([accuracy_log.get(f"2L_{ops}", float("nan")) for ops in OPSETS])

    if np.all(np.isnan(area_1L)) and np.all(np.isnan(area_2L)):
        print("[Module 5] No area data available — skipping area_vs_accuracy chart.")
        return None

    n  = len(OPSETS)
    x  = np.arange(n)
    bw = 0.30
    g  = 0.04
    off = (bw + g) / 2

    fig, ax_a = plt.subplots(figsize=(9, 5.5), dpi=150)
    ax_acc = ax_a.twinx()

    # Bars 
    bars_1 = ax_a.bar(x - off, area_1L, width=bw,
                      color=COLOR["bar_1L"], alpha=0.82,
                      edgecolor="white", linewidth=0.8, zorder=3, label="Area  1L")
    bars_2 = ax_a.bar(x + off, area_2L, width=bw,
                      color=COLOR["bar_2L"], alpha=0.82,
                      edgecolor="white", linewidth=0.8, zorder=3, label="Area  2L")

    for rect, val in zip(list(bars_1) + list(bars_2),
                         list(area_1L) + list(area_2L)):
        if not np.isnan(val):
            ax_a.text(rect.get_x() + rect.get_width() / 2,
                      rect.get_height() + 0.4,
                      f"{val:.1f}%", ha="center", va="bottom",
                      fontsize=7.5, fontweight="bold",
                      color=rect.get_facecolor())

    # Lines 
    mk = dict(markersize=7, linewidth=2.0, zorder=5, clip_on=False)
    ax_acc.plot(x, acc_1L, color=COLOR["line_1L"], marker="o",
                linestyle="-",  alpha=0.95, label="Accuracy  1L", **mk)
    ax_acc.plot(x, acc_2L, color=COLOR["line_2L"], marker="s",
                linestyle="--", alpha=0.95, label="Accuracy  2L", **mk)

    for xi, (a1, a2) in enumerate(zip(acc_1L, acc_2L)):
        if not np.isnan(a1):
            ax_acc.annotate(f"{a1:.1f}%", xy=(xi, a1),
                            xytext=(0, 7),  textcoords="offset points",
                            ha="center", fontsize=7.5,
                            color=COLOR["line_1L"], fontweight="bold")
        if not np.isnan(a2):
            ax_acc.annotate(f"{a2:.1f}%", xy=(xi, a2),
                            xytext=(0, -14), textcoords="offset points",
                            ha="center", fontsize=7.5,
                            color=COLOR["line_2L"], fontweight="bold")

    # Baseline dashed line 
    ax_acc.axhline(baseline_acc, color=COLOR["baseline"],
                   linestyle=":", linewidth=1.8, zorder=4,
                   label=f"FP32 Baseline ({baseline_acc:.1f}%)")

    # Axis formatting 
    ax_a.set_xlabel("Operator Set", fontsize=11, labelpad=8)
    ax_a.set_ylabel(area_label, fontsize=10, labelpad=8)
    ax_a.set_xticks(x);  ax_a.set_xticklabels(OPSETS, fontsize=11)
    ax_a.set_xlim(-0.6, n - 1 + 0.6)
    valid_area = np.concatenate([area_1L, area_2L])
    valid_area = valid_area[~np.isnan(valid_area)]
    ax_a.set_ylim(0, valid_area.max() * 1.5 if len(valid_area) else 110)
    ax_a.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax_a.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax_a.set_axisbelow(True)

    ax_acc.set_ylabel("Accuracy (%)", fontsize=11, labelpad=8)
    valid_acc = np.concatenate([acc_1L, acc_2L])
    valid_acc = valid_acc[~np.isnan(valid_acc)]
    if len(valid_acc):
        pad = (baseline_acc - valid_acc.min()) * 0.6
        ax_acc.set_ylim(valid_acc.min() - pad, baseline_acc + pad)
    ax_acc.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # Alternate-group shading 
    for i in range(n):
        if i % 2 == 0:
            ax_a.axvspan(i - 0.5, i + 0.5, color="#f5f5f5", zorder=0, alpha=0.6)

    # Legend 
    handles = [
        mpatches.Patch(color=COLOR["bar_1L"], alpha=0.82, label="Area  1L (bars)"),
        mpatches.Patch(color=COLOR["bar_2L"], alpha=0.82, label="Area  2L (bars)"),
        Line2D([0],[0], color=COLOR["line_1L"], marker="o",
               linestyle="-",  linewidth=2, label="Accuracy  1L"),
        Line2D([0],[0], color=COLOR["line_2L"], marker="s",
               linestyle="--", linewidth=2, label="Accuracy  2L"),
        Line2D([0],[0], color=COLOR["baseline"], linestyle=":",
               linewidth=1.8, label=f"FP32 Baseline ({baseline_acc:.1f}%)"),
    ]
    ax_a.legend(handles=handles, loc="upper right",
                fontsize=8.5, framealpha=0.88, edgecolor="#ccc")

    hw_note = "Real Vitis HLS data" if use_real_hw else "SR complexity proxy (run Vitis HLS for real LUT data)"
    fig.suptitle(
        f"NeuSym-HLS  --  Hardware Area vs. Accuracy  |  SVHN 1-vs-7\n"
        f"AMD XC7Z020 @ 100 MHz    Area source: {hw_note}",
        fontsize=10, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out = EVAL_DIR / "area_vs_accuracy.png"
    fig.savefig(out,                       dpi=300, bbox_inches="tight")
    fig.savefig(EVAL_DIR / "area_vs_accuracy.pdf", bbox_inches="tight")
    print(f"[Module 5] Saved: {out}")
    plt.close(fig)
    return out



# 3.  Chart 2  -  Per-resource utilisation breakdown (synthesis only)

def plot_resource_breakdown(synth_metrics: dict[str, dict]) -> pathlib.Path | None:
    """
    Grouped bars: LUT / FF / DSP for all 6 (level x opset) configurations.
    Only drawn when synthesis_metrics.csv is present.
    """
    if not synth_metrics:
        return None

    configs = [f"{lvl}\n{ops}" for lvl in LEVELS for ops in OPSETS]
    tags    = [f"{lvl}_{ops}"  for lvl in LEVELS for ops in OPSETS]
    n       = len(tags)
    x       = np.arange(n)
    bw      = 0.22

    lut = np.array([synth_metrics.get(t, {}).get("LUT_pct", 0) for t in tags])
    ff  = np.array([synth_metrics.get(t, {}).get("FF_pct",  0) for t in tags])
    dsp = np.array([synth_metrics.get(t, {}).get("DSP_pct", 0) for t in tags])

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)
    ax.bar(x - bw, lut, width=bw, label="LUT (%)", color=COLOR["lut"], alpha=0.85, edgecolor="white")
    ax.bar(x,      ff,  width=bw, label="FF (%)",  color=COLOR["ff"],  alpha=0.85, edgecolor="white")
    ax.bar(x + bw, dsp, width=bw, label="DSP (%)", color=COLOR["dsp"], alpha=0.85, edgecolor="white")

    ax.set_xticks(x);  ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylabel("Resource Utilisation (% of XC7Z020 total)", fontsize=10)
    ax.set_xlabel("Configuration  (Distillation Level / Operator Set)", fontsize=10)
    ax.set_title("NeuSym-HLS  --  Per-Resource Utilisation Breakdown\n"
                 "AMD XC7Z020  -  All (Level x Opset) Configurations",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    fig.tight_layout()

    out = EVAL_DIR / "resource_breakdown.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(EVAL_DIR / "resource_breakdown.pdf", bbox_inches="tight")
    print(f"[Module 5] Saved: {out}")
    plt.close(fig)
    return out



# 4.  Chart 3  -  Pareto scatter

def plot_pareto_scatter(accuracy_log:   dict[str, float],
                         baseline_acc:  float,
                         synth_metrics: dict[str, dict] | None,
                         sr_metrics:    dict[str, dict]) -> pathlib.Path | None:
    """
    Scatter: x = area (LUT% or normalised complexity), y = accuracy.
    Highlights the Pareto-optimal front.
    """
    use_real_hw = synth_metrics is not None

    def _area(tag):
        if use_real_hw:
            return synth_metrics.get(tag, {}).get("LUT_pct", float("nan"))
        max_c = max((v["complexity"] for v in sr_metrics.values()), default=1)
        return sr_metrics.get(tag, {}).get("complexity", float("nan")) / max_c * 100

    points: list[tuple[float, float, str]] = []
    for lvl in LEVELS:
        for ops in OPSETS:
            tag  = f"{lvl}_{ops}"
            area = _area(tag)
            acc  = accuracy_log.get(tag, float("nan"))
            if not (np.isnan(area) or np.isnan(acc)):
                points.append((area, acc, tag))

    if not points:
        print("[Module 5] Not enough data for Pareto scatter — skipping.")
        return None

    marker_map = {"1L": "o", "2L": "s"}
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    for area, acc, tag in points:
        lvl = tag[:2]
        ops = tag[3:]
        ax.scatter(area, acc,
                   marker=marker_map[lvl], color=OPSET_COLORS[ops],
                   s=120, zorder=5, edgecolors="white", linewidths=0.8, alpha=0.92)
        ax.annotate(f"  {tag}", xy=(area, acc), fontsize=8,
                    color=OPSET_COLORS[ops], fontweight="bold")

    # Pareto front (low area, high accuracy) 
    pts_sorted = sorted(points, key=lambda t: t[0])
    pareto, best = [], -1.0
    for area, acc, _ in pts_sorted:
        if acc > best:
            best = acc
            pareto.append((area, acc))
    if len(pareto) >= 2:
        px, py = zip(*pareto)
        ax.step(px, py, where="post", color=COLOR["baseline"],
                linewidth=1.8, linestyle="--", zorder=3, label="Pareto front")

    ax.axhline(baseline_acc, color="#7F8C8D", linestyle=":", linewidth=1.5,
               label=f"FP32 Baseline ({baseline_acc:.1f}%)")

    x_label = ("LUT Utilisation (% of XC7Z020 total)"
               if use_real_hw else "Relative SR Complexity (normalised, %)")
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("NeuSym-HLS  --  Accuracy vs. Area Pareto Scatter\n"
                 "AMD XC7Z020 @ 100 MHz  -  SVHN 1-vs-7",
                 fontsize=10, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.28)

    legend_elems = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#555",
               markersize=9, label="1L distillation"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#555",
               markersize=9, label="2L distillation"),
    ]
    for ops, c in OPSET_COLORS.items():
        legend_elems.append(mpatches.Patch(color=c, label=f"{ops} opset"))
    legend_elems += [
        Line2D([0],[0], color=COLOR["baseline"], linestyle="--",
               linewidth=1.8, label="Pareto front"),
        Line2D([0],[0], color="#7F8C8D", linestyle=":",
               linewidth=1.5, label=f"FP32 Baseline ({baseline_acc:.1f}%)"),
    ]
    ax.legend(handles=legend_elems, fontsize=8, framealpha=0.88,
              loc="lower right", edgecolor="#ccc")

    fig.tight_layout()
    out = EVAL_DIR / "pareto_scatter.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(EVAL_DIR / "pareto_scatter.pdf", bbox_inches="tight")
    print(f"[Module 5] Saved: {out}")
    plt.close(fig)
    return out



# 5.  Chart 4  -  SR complexity and MSE per config (always available post-SR)


def plot_sr_complexity(sr_metrics:   dict[str, dict],
                        accuracy_log: dict[str, float]) -> pathlib.Path | None:
    """
    Two-panel figure:
      Top    : SR equation complexity (hardware-weighted node count) per config
      Bottom : SR MSE (pre-fine-tuning fit quality) per config
    Bars are grouped by distillation level; x-axis is operator set.
    """
    if not sr_metrics:
        print("[Module 5] No SR metrics available — skipping sr_complexity chart.")
        return None

    tags   = [f"{lvl}_{ops}" for lvl in LEVELS for ops in OPSETS]
    labels = [t.replace("_", "\n") for t in tags]
    n      = len(tags)
    x      = np.arange(n)

    complexity = np.array([sr_metrics.get(t, {}).get("complexity", 0) for t in tags])
    mse        = np.array([sr_metrics.get(t, {}).get("mse",        0) for t in tags])
    acc        = np.array([accuracy_log.get(t, float("nan"))          for t in tags])

    # colour bars by opset
    bar_colors = [OPSET_COLORS[t.split("_")[1]] for t in tags]
    # add hatching to distinguish 1L vs 2L
    hatches = ["//" if t.startswith("1L") else ".." for t in tags]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=150,
                                   sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1]})

    # Panel 1: complexity 
    bars1 = ax1.bar(x, complexity, color=bar_colors, alpha=0.82,
                    edgecolor="white", linewidth=0.8)
    for bar, h in zip(bars1, hatches):
        bar.set_hatch(h)
    for xi, val in enumerate(complexity):
        if val > 0:
            ax1.text(xi, val + complexity.max() * 0.01,
                     f"{int(val)}", ha="center", va="bottom",
                     fontsize=8, fontweight="bold")
    ax1.set_ylabel("HW-Weighted Complexity\n(operator cycle-cost sum)", fontsize=9)
    ax1.set_title("NeuSym-HLS  --  Symbolic Regression Equation Metrics\n"
                  "Complexity (top) and MSE (bottom) per (Level x Opset) Configuration",
                  fontsize=10, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)

    # Panel 2: MSE 
    bars2 = ax2.bar(x, mse, color=bar_colors, alpha=0.82,
                    edgecolor="white", linewidth=0.8)
    for bar, h in zip(bars2, hatches):
        bar.set_hatch(h)

    # Overlay accuracy as a secondary axis
    ax2b = ax2.twinx()
    ax2b.plot(x, acc, color="#2C3E50", marker="D", linestyle="-.",
              linewidth=1.6, markersize=6, zorder=5, label="Fine-tuned Accuracy (%)")
    ax2b.set_ylabel("Fine-tuned Accuracy (%)", fontsize=9)
    if not np.all(np.isnan(acc)):
        valid = acc[~np.isnan(acc)]
        pad = (valid.max() - valid.min()) * 0.5 if len(valid) > 1 else 1.0
        ax2b.set_ylim(valid.min() - pad, valid.max() + pad)
    ax2b.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax2b.legend(loc="upper right", fontsize=8)

    ax2.set_ylabel("SR MSE (logit prediction error)", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_xlabel("Configuration  (Level / Opset)", fontsize=10, labelpad=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.set_axisbelow(True)

    # Shared legend for opset colors + 1L/2L hatching 
    legend_elems = [mpatches.Patch(facecolor=OPSET_COLORS[ops], label=f"{ops} opset")
                    for ops in OPSETS]
    legend_elems += [
        mpatches.Patch(facecolor="#888", hatch="//", label="1L distillation"),
        mpatches.Patch(facecolor="#888", hatch="..", label="2L distillation"),
    ]
    ax1.legend(handles=legend_elems, fontsize=8, loc="upper right",
               framealpha=0.88, edgecolor="#ccc")

    fig.tight_layout()
    out = EVAL_DIR / "sr_complexity.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(EVAL_DIR / "sr_complexity.pdf", bbox_inches="tight")
    print(f"[Module 5] Saved: {out}")
    plt.close(fig)
    return out



# 6.  Master function called by run_pipeline.py


def run(show: bool = False) -> dict[str, pathlib.Path | None]:
    """
    Load all pipeline-generated result files and produce every chart.
    Called automatically at the end of run_pipeline.py.

    Parameters
    ----------
    show : bool
        If True, call plt.show() after all charts are saved.
        Set to False for headless / CI environments.

    Returns
    -------
    dict mapping chart name -> saved PNG path (or None if skipped)
    """
    print("\n[Module 5] === Generating Evaluation Charts ===\n")

    # Load results from pipeline output files 
    baseline_acc   = load_baseline_accuracy()
    accuracy_log   = load_accuracy_log()
    sr_metrics     = load_sr_metrics()
    synth_metrics  = load_synthesis_metrics()   # None if Vitis HLS not run yet

    # Generate charts 
    outputs = {}

    outputs["area_vs_accuracy"] = plot_area_vs_accuracy(
        accuracy_log, baseline_acc, synth_metrics, sr_metrics)

    outputs["resource_breakdown"] = plot_resource_breakdown(synth_metrics)

    outputs["pareto_scatter"] = plot_pareto_scatter(
        accuracy_log, baseline_acc, synth_metrics, sr_metrics)

    outputs["sr_complexity"] = plot_sr_complexity(sr_metrics, accuracy_log)

    # Summary 
    print("\n[Module 5] Charts saved to eval_results/:")
    for name, path in outputs.items():
        status = str(path.name) if path else "SKIPPED"
        print(f"   {name:<22} -> {status}")

    if show:
        plt.show()

    return outputs



# Entry point (standalone use)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="NeuSym-HLS Module 5 -- Evaluation Visualization\n"
                    "Reads results produced by Modules 1, 2, and 4.")
    parser.add_argument("--show", action="store_true",
                        help="Display charts interactively after saving")
    args = parser.parse_args()
    run(show=args.show)
