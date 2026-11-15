"""Render all six C1 targets at every frozen observation budget."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / "results/l1-mastery/2026-11-14-C"
    result = json.loads((directory / "phase-C-summary.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "dovis-mastery-C-43632332293"})
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True, sharey=True)
    specifications = (("Understanding", "comprehension_accuracy", "#175cd3", "-", 2),
                      ("Free generation", "valid_generation_accuracy", "#dc6803", "-", 2),
                      ("Role", "role_reversal", "#53b1fd", ":", 1.4),
                      ("Color", "color_only", "#7f56d9", "--", 1.4),
                      ("Shape", "shape_only", "#039855", "--", 1.4),
                      ("Binding", "binding_swap", "#667085", ":", 1.4))
    for axis, language, label in zip(axes.flat, ("ko", "ja", "en", "es"),
                                     ("Korean", "Japanese", "English", "Spanish")):
        rows = sorted((row for row in result["rows"] if row["condition"] == "C1"
                       and row["language"] == language), key=lambda row: row["step"])
        for name, metric, color, style, width in specifications:
            values = [row["dev"][metric] if metric.endswith("accuracy")
                      else row["contrasts"][metric]["accuracy"] for row in rows]
            axis.plot([row["step"] for row in rows], values, color=color,
                      linestyle=style, linewidth=width, marker="o", markersize=3.5, label=name)
        axis.axhline(.9, color="#d0d5dd", linestyle=(0, (4, 4)), linewidth=1)
        axis.set_title(label, loc="left", fontsize=12, fontweight="bold")
        axis.set_ylim(-.03, 1.04)
        axis.set_xlim(-180, 9800)
        axis.set_xticks((0, 2400, 4800, 9600), ("0", "2,400", "4,800", "9,600"))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", color="#eaecf0", linewidth=.8)
    for axis in axes[-1]:
        axis.set_xlabel("L1 updates")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(.06, .925),
                  ncol=3, frameon=False, columnspacing=2.5)
    figure.suptitle("C1 native scene learning at fixed budgets", x=.07, ha="left",
                    fontsize=18, fontweight="bold")
    figure.text(.07, .028, "One L1 seed; shared image-only prior. Binary chance: 50%; original choices: 25%. Dashed horizontal line: 90% target.",
                color="#475467", fontsize=9)
    figure.subplots_adjust(left=.07, right=.98, bottom=.12, top=.77, hspace=.32, wspace=.15)
    figure.savefig(directory / "curves-C1.svg", metadata={"Date": None})
    figure.savefig("/tmp/dovis-mastery-c-preview.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
