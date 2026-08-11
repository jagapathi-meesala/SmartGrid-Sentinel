# =============================================================================
# scripts/generate_paper_figures.py
# Generates clean, publication-ready static figures from REAL result JSON
# files (results/baseline_metrics.json, fl_metrics.json, twin_events.json,
# ara_actions.json) for direct embedding in the IEEE paper.
#
# Every number plotted here comes from an actual run. The uniform-loss
# comparison numbers (pre-class-weighting) are hardcoded because that run
# was not saved to its own JSON file -- they are the real, verified numbers
# reported earlier (98.72% accuracy / 48.63% recall / 89.05% precision),
# not fabricated for this figure.
#
# Run: python scripts/generate_paper_figures.py
# Output: results/figures/*.png (300 DPI, white background, IEEE-column width)
# =============================================================================

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from config.settings import RESULTS_DIR, CLASS_NAMES, NORMAL_LABEL_NAME

FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "font.family": "serif",
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "figure.dpi": 150, "savefig.dpi": 300,
})

IEEE_COL_WIDTH = 3.45  # inches, single-column IEEE figure width


def load(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"[SKIP] {name} not found -- run the corresponding script first.")
        return None
    with open(path) as f:
        return json.load(f)


# ── Figure 1: Confusion matrix (class-weighted, real) ───────────────────────
def fig_confusion_matrix(baseline):
    if not baseline:
        return
    cm = np.array(baseline["final_metrics"]["confusion_matrix"])
    names = list(baseline["class_names"].values())
    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, IEEE_COL_WIDTH * 0.85))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=0)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(names)):
        for j in range(len(names)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                     color=color, fontsize=9)
    ax.set_title("Confusion Matrix — Class-Weighted Centralized Model", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_confusion_matrix.png"), bbox_inches="tight")
    plt.close(fig)
    print("[OK] fig_confusion_matrix.png")


# ── Figure 2: Uniform vs. class-weighted loss comparison ────────────────────
def fig_class_weighting_comparison():
    # Real, verified numbers from the actual uniform-loss and class-weighted
    # runs (see README / digest Table I). Not saved to JSON originally, but
    # not fabricated -- these are the confirmed run outputs.
    metrics = ["Accuracy", "Attack\nRecall", "Attack\nPrecision"]
    uniform = [98.72, 48.63, 89.05]
    weighted = [98.01, 80.38, 53.54]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, IEEE_COL_WIDTH * 0.8))
    b1 = ax.bar(x - width/2, uniform, width, label="Uniform loss", color="#94a3b8")
    b2 = ax.bar(x + width/2, weighted, width, label="Class-weighted", color="#2563eb")
    ax.set_ylabel("%"); ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8, loc="upper right")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width()/2, h),
                        ha="center", va="bottom", fontsize=7)
    ax.set_title("Effect of Class-Weighted Loss on Detection Performance", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_class_weighting.png"), bbox_inches="tight")
    plt.close(fig)
    print("[OK] fig_class_weighting.png")


# ── Figure 3: FL convergence over communication rounds ───────────────────────
def fig_fl_convergence(fl):
    if not fl or not fl.get("rounds"):
        return
    rounds = [r["round"] for r in fl["rounds"]]
    acc = [r["accuracy"] * 100 for r in fl["rounds"]]
    f1 = [r["f1"] * 100 for r in fl["rounds"]]

    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH, IEEE_COL_WIDTH * 0.75))
    ax.plot(rounds, acc, marker="o", ms=3.5, label="Accuracy", color="#2563eb")
    ax.plot(rounds, f1, marker="s", ms=3.5, label="F1 (weighted)", color="#16a34a")
    ax.set_xlabel("Communication round"); ax.set_ylabel("%")
    ax.set_xticks(rounds)
    ax.legend(fontsize=8)
    ax.set_title("Federated Learning Convergence (FedAvg, 3 real clients)", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_fl_convergence.png"), bbox_inches="tight")
    plt.close(fig)
    print("[OK] fig_fl_convergence.png")


# ── Figure 4: Digital twin risk score per tick vs. calibrated threshold ─────
def fig_twin_risk_timeline(twin_events):
    if not twin_events:
        return
    by_tick = {}
    for e in twin_events:
        by_tick.setdefault(e["tick"], []).append(e["risk"])
    ticks = sorted(by_tick)
    max_risk = [max(by_tick[t]) for t in ticks]
    threshold = twin_events[0]["dynamic_threshold"]
    injected_ticks = sorted({e["tick"] for e in twin_events if e.get("was_injected")})

    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH * 1.6, IEEE_COL_WIDTH * 0.7))
    ax.plot(ticks, max_risk, color="#2563eb", lw=1.2, label="Max risk this tick")
    ax.axhline(threshold, color="#dc2626", ls="--", lw=1,
               label=f"Calibrated threshold ({threshold:.3f})")
    for t in injected_ticks:
        ax.axvline(t, color="#f59e0b", ls=":", lw=1, alpha=0.7)
    ax.scatter(injected_ticks, [max_risk[ticks.index(t)] for t in injected_ticks],
               color="#f59e0b", zorder=5, s=25, label="Injected attack tick")
    ax.set_xlabel("Simulation tick"); ax.set_ylabel("Risk score")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Digital Twin: Real-Time Risk Score vs. Calibrated Threshold", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_twin_risk_timeline.png"), bbox_inches="tight")
    plt.close(fig)
    print("[OK] fig_twin_risk_timeline.png")


# ── Figure 5: ARA isolate/restore timeline ───────────────────────────────────
def fig_ara_timeline(ara_actions):
    if not ara_actions:
        return
    subs = sorted({a["substation"] for a in ara_actions})
    fig, ax = plt.subplots(figsize=(IEEE_COL_WIDTH * 1.6, IEEE_COL_WIDTH * 0.55))
    y_pos = {s: i for i, s in enumerate(subs)}
    for a in ara_actions:
        color = "#dc2626" if a["to_state"] == "isolated" else "#16a34a"
        marker = "s" if a["to_state"] == "isolated" else "o"
        ax.scatter(a["tick"], y_pos[a["substation"]], color=color, marker=marker, s=60, zorder=5)
    ax.set_yticks(list(y_pos.values())); ax.set_yticklabels(list(y_pos.keys()))
    ax.set_xlabel("Simulation tick")
    ax.set_title("Autonomous Response Agent: Isolation / Restoration Actions", fontsize=9)
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#dc2626", markersize=8, label="Isolated"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#16a34a", markersize=8, label="Restored"),
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_ara_timeline.png"), bbox_inches="tight")
    plt.close(fig)
    print("[OK] fig_ara_timeline.png")


def main():
    baseline = load("baseline_metrics.json")
    fl = load("fl_metrics.json")
    twin_events = load("twin_events.json")
    ara_actions = load("ara_actions.json")

    fig_confusion_matrix(baseline)
    fig_class_weighting_comparison()
    fig_fl_convergence(fl)
    fig_twin_risk_timeline(twin_events)
    fig_ara_timeline(ara_actions)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()