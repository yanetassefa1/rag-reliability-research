"""
Generate figures from evaluation results.
Run after evaluate.py has produced results/summary.json
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

os.makedirs("plots", exist_ok=True)

with open("results/summary.json") as f:
    summary = json.load(f)

CONDITIONS_ORDER = [
    "clean_level0",
    "irrelevant_level1",
    "irrelevant_level3",
    "contradictory_level1",
    "contradictory_level2",
    "misleading_level1",
    "misleading_level2",
]

LABELS = {
    "clean_level0": "Clean\n(baseline)",
    "irrelevant_level1": "Irrelevant\n(1 doc)",
    "irrelevant_level3": "Irrelevant\n(3 docs)",
    "contradictory_level1": "Contradictory\n(1 doc)",
    "contradictory_level2": "Contradictory\n(2 docs)",
    "misleading_level1": "Misleading\n(1 doc)",
    "misleading_level2": "Misleading\n(2 docs)",
}

COLORS = {
    "accuracy_rate": "#378ADD",
    "hallucination_rate": "#E24B4A",
    "grounding_rate": "#639922",
    "abstention_rate": "#BA7517",
}

conditions = [c for c in CONDITIONS_ORDER if c in summary]
x = np.arange(len(conditions))
x_labels = [LABELS[c] for c in conditions]


# ── Figure 1: Accuracy vs Hallucination across noise conditions ───────────────
fig, ax = plt.subplots(figsize=(11, 5))
acc = [summary[c]["accuracy_rate"] for c in conditions]
hal = [summary[c]["hallucination_rate"] for c in conditions]
width = 0.35
bars1 = ax.bar(x - width/2, acc, width, color=COLORS["accuracy_rate"], label="Accuracy rate", alpha=0.85)
bars2 = ax.bar(x + width/2, hal, width, color=COLORS["hallucination_rate"], label="Hallucination rate", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_ylabel("Rate", fontsize=11)
ax.set_title("Accuracy vs. Hallucination Rate by Noise Condition", fontsize=12, fontweight="bold", pad=12)
ax.set_ylim(0, 1.15)
ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}", ha="center", fontsize=8, color="#185FA5")
for bar in bars2:
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}", ha="center", fontsize=8, color="#A32D2D")
plt.tight_layout()
plt.savefig("plots/fig1_accuracy_vs_hallucination.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/fig1_accuracy_vs_hallucination.png")


# ── Figure 2: Abstention rate across conditions ───────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.5))
abst = [summary[c]["abstention_rate"] for c in conditions]
bars = ax.bar(x, abst, color=COLORS["abstention_rate"], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_ylabel("Abstention rate", fontsize=11)
ax.set_title("Model Abstention Rate by Noise Condition", fontsize=12, fontweight="bold", pad=12)
ax.set_ylim(0, 1.1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}", ha="center", fontsize=9, color="#633806")
plt.tight_layout()
plt.savefig("plots/fig2_abstention_rate.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/fig2_abstention_rate.png")


# ── Figure 3: All metrics heatmap ─────────────────────────────────────────────
metrics = ["accuracy_rate", "hallucination_rate", "grounding_rate", "abstention_rate"]
metric_labels = ["Accuracy", "Hallucination", "Grounding", "Abstention"]
data = np.array([[summary[c][m] for c in conditions] for m in metrics])
fig, ax = plt.subplots(figsize=(11, 4))
im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(conditions)))
ax.set_xticklabels(x_labels, fontsize=8.5)
ax.set_yticks(range(len(metrics)))
ax.set_yticklabels(metric_labels, fontsize=10)
ax.set_title("Reliability Metrics Across Noise Conditions (heatmap)", fontsize=12, fontweight="bold", pad=12)
for i in range(len(metrics)):
    for j in range(len(conditions)):
        val = data[i, j]
        color = "black" if 0.3 < val < 0.7 else ("white" if val <= 0.3 else "black")
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color, fontweight="500")
plt.colorbar(im, ax=ax, shrink=0.8, label="Rate")
plt.tight_layout()
plt.savefig("plots/fig3_metrics_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/fig3_metrics_heatmap.png")


# ── Figure 4: Accuracy degradation relative to baseline ──────────────────────
baseline_acc = summary.get("clean_level0", {}).get("accuracy_rate", 1.0)
degradation = [baseline_acc - summary[c]["accuracy_rate"] for c in conditions]
colors_deg = ["#378ADD" if d <= 0 else "#E24B4A" for d in degradation]
fig, ax = plt.subplots(figsize=(11, 4.5))
bars = ax.bar(x, degradation, color=colors_deg, alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_ylabel("Accuracy drop from baseline", fontsize=11)
ax.set_title("Accuracy Degradation Relative to Clean Baseline", fontsize=12, fontweight="bold", pad=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
patch1 = mpatches.Patch(color="#E24B4A", alpha=0.85, label="Performance degradation")
patch2 = mpatches.Patch(color="#378ADD", alpha=0.85, label="No degradation / improvement")
ax.legend(handles=[patch1, patch2], fontsize=9)
for bar, d in zip(bars, degradation):
    offset = 0.01 if d >= 0 else -0.04
    ax.text(bar.get_x() + bar.get_width()/2, d + offset, f"{d:+.2f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("plots/fig4_accuracy_degradation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plots/fig4_accuracy_degradation.png")

print("\nAll figures saved to plots/")
