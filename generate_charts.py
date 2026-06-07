"""
generate_charts.py — Generate result visualisation charts for README
======================================================================
Produces:
  img/results_e1_e9.png   — MRR bar chart for E1–E9 system experiments
  img/sensitivity.png     — Chunk parameter sensitivity comparison

Run: source venv/bin/activate && python generate_charts.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EVAL_DIR     = PROJECT_ROOT / "data" / "eval"
IMG_DIR      = PROJECT_ROOT / "img"
IMG_DIR.mkdir(exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})

PALETTE = {
    "mrr":      "#4f46e5",  # indigo
    "recall5":  "#10b981",  # green
    "recall10": "#f59e0b",  # amber
}


# ── Chart 1: System Experiments E1–E9 ────────────────────────────────────────

E1_E9 = [
    ("E1\nC1 no OCR",         0.4478, 0.5952, 0.6786),
    ("E2\nC1+OCR",            0.6536, 0.7857, 0.8452),
    ("E3\nC1+OCR+BM25",       0.6564, 0.7976, 0.8571),
    ("E4\nC2 no OCR",         0.4059, 0.5595, 0.7024),
    ("E5\nC2+OCR",            0.6066, 0.7262, 0.7857),
    ("E6\nC2+OCR+BM25",       0.6148, 0.7619, 0.8214),
    ("E7\nC3 no OCR",         0.4487, 0.6429, 0.6786),
    ("E8\nC3+OCR",            0.8200, 0.9643, 0.9643),
    ("E9\nC3+OCR+BM25 ★",    0.8259, 0.9524, 0.9643),
]

labels   = [x[0] for x in E1_E9]
mrr      = [x[1] for x in E1_E9]
recall5  = [x[2] for x in E1_E9]
recall10 = [x[3] for x in E1_E9]

x    = np.arange(len(labels))
w    = 0.26

fig, ax = plt.subplots(figsize=(13, 5.5))
fig.patch.set_facecolor("#fafafa")
ax.set_facecolor("#fafafa")

b1 = ax.bar(x - w,     mrr,      w, label="MRR",       color=PALETTE["mrr"],      alpha=0.9, zorder=3)
b2 = ax.bar(x,         recall5,  w, label="Recall@5",  color=PALETTE["recall5"],  alpha=0.9, zorder=3)
b3 = ax.bar(x + w,     recall10, w, label="Recall@10", color=PALETTE["recall10"], alpha=0.9, zorder=3)

# Annotate best experiment (E9)
ax.bar(x[-1] - w,     mrr[-1],      w, color=PALETTE["mrr"],      edgecolor="#1e1b4b", linewidth=2, zorder=4)
ax.bar(x[-1],         recall5[-1],  w, color=PALETTE["recall5"],  edgecolor="#064e3b", linewidth=2, zorder=4)
ax.bar(x[-1] + w,     recall10[-1], w, color=PALETTE["recall10"], edgecolor="#78350f", linewidth=2, zorder=4)

# Value labels on top of each bar
for bar in [*b1, *b2, *b3]:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008, f"{h:.2f}",
            ha="center", va="bottom", fontsize=6.5, color="#374151")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", fontsize=11)
ax.set_xlabel("Experiment", fontsize=11)
ax.set_title("System Experiment Results: MRR & Recall@K  (84 human-annotated queries)",
             fontsize=12, fontweight="bold", pad=14)

legend = ax.legend(
    handles=[
        mpatches.Patch(color=PALETTE["mrr"],      label="MRR"),
        mpatches.Patch(color=PALETTE["recall5"],  label="Recall@5"),
        mpatches.Patch(color=PALETTE["recall10"], label="Recall@10"),
    ],
    loc="upper left", fontsize=10, framealpha=0.6
)

# Shade C3 region
ax.axvspan(6 - 0.5, 8 + 0.5, alpha=0.06, color="#4f46e5", label="C3 region")
ax.text(7.0, 1.07, "C3 slide-boundary\n(best strategy)", ha="center",
        fontsize=8.5, color="#4f46e5", fontstyle="italic")

# Vertical separators between strategy groups
for pos in [2.5, 5.5]:
    ax.axvline(pos, color="#9ca3af", linewidth=1, linestyle="--", alpha=0.7)
ax.text(1.0,  1.10, "C1", ha="center", fontsize=9, color="#6b7280", fontweight="bold")
ax.text(4.0,  1.10, "C2", ha="center", fontsize=9, color="#6b7280", fontweight="bold")

out = IMG_DIR / "results_e1_e9.png"
fig.tight_layout()
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved → {out}")


# ── Chart 2: Chunk Parameter Sensitivity ─────────────────────────────────────

SENSITIVITY = [
    ("C2\n150w", 0.4527, 0.6310, 0.7381),
    ("C2\n200w", 0.4059, 0.5595, 0.7024),
    ("C2\n250w", 0.3762, 0.5238, 0.6429),
    ("C3\nt=0.25", 0.4494, 0.6429, 0.6905),
    ("C3\nt=0.30", 0.4487, 0.6429, 0.6786),
    ("C3\nt=0.40", 0.4467, 0.6548, 0.6905),
]

labels2   = [x[0] for x in SENSITIVITY]
mrr2      = [x[1] for x in SENSITIVITY]
recall52  = [x[2] for x in SENSITIVITY]
recall102 = [x[3] for x in SENSITIVITY]

x2 = np.arange(len(labels2))
fig2, ax2 = plt.subplots(figsize=(9, 4.5))
fig2.patch.set_facecolor("#fafafa")
ax2.set_facecolor("#fafafa")

ax2.bar(x2 - w, mrr2,      w, label="MRR",       color=PALETTE["mrr"],      alpha=0.9, zorder=3)
ax2.bar(x2,     recall52,  w, label="Recall@5",  color=PALETTE["recall5"],  alpha=0.9, zorder=3)
ax2.bar(x2 + w, recall102, w, label="Recall@10", color=PALETTE["recall10"], alpha=0.9, zorder=3)

ax2.axvspan(-0.5, 2.5, alpha=0.05, color="#10b981")
ax2.axvspan(2.5,  5.5, alpha=0.05, color="#4f46e5")
ax2.text(1.0, 0.82, "C2 word-count", ha="center", fontsize=8.5, color="#065f46", fontstyle="italic")
ax2.text(4.0, 0.82, "C3 slide-boundary", ha="center", fontsize=8.5, color="#4f46e5", fontstyle="italic")
ax2.axvline(2.5, color="#9ca3af", linewidth=1, linestyle="--", alpha=0.7)

ax2.set_xticks(x2)
ax2.set_xticklabels(labels2, fontsize=9)
ax2.set_ylim(0, 0.9)
ax2.set_ylabel("Score", fontsize=11)
ax2.set_xlabel("Chunking Variant", fontsize=11)
ax2.set_title("Chunk Parameter Sensitivity (OCR=OFF, BM25=OFF — pure chunking effect)",
              fontsize=11, fontweight="bold", pad=12)
ax2.legend(fontsize=10, framealpha=0.6)

out2 = IMG_DIR / "sensitivity_chart.png"
fig2.tight_layout()
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close(fig2)
print(f"Saved → {out2}")

print("\nDone! Add to README.md:")
print("  ![Results](img/results_e1_e9.png)")
print("  ![Sensitivity](img/sensitivity_chart.png)")
