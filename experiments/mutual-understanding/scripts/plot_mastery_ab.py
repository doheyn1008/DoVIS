"""Render the frozen A/B results; optional matplotlib is needed only here."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / "results/l1-mastery/2026-11-14-AB"
    phases = {phase: json.loads((directory / f"phase-{phase}-summary.json").read_text())
              for phase in ("A", "B")}
    languages = ("ko", "ja", "en", "es")
    labels = ("Korean", "Japanese", "English", "Spanish")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "dovis-mastery-AB-43630516311"})
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=True)
    panels = (("A", "Original four-choice understanding", "comprehension_accuracy"),
              ("A", "Shape-only binary diagnostic", "shape_only"),
              ("B", "Original four-choice understanding", "comprehension_accuracy"),
              ("B", "Valid free generation", "valid_generation_accuracy"))
    for axis, (phase, title, metric) in zip(axes.flat, panels):
        rows = [row for row in phases[phase]["rows"] if row["step"] == 2400]
        colors = ("#667085", "#1570ef" if phase == "A" else "#d45a0b")
        for index, language in enumerate(languages):
            for arm in (0, 1):
                values = [row["contrasts"]["shape_only"]["accuracy"] if metric == "shape_only"
                          else row["dev"][metric] for row in rows
                          if row["language"] == language and row["condition"] == phase + str(arm)]
                x = index + (-.14 if arm == 0 else .14)
                if phase == "A":
                    for seed, value in enumerate(values):
                        axis.scatter(x + (seed - 1) * .026, value, s=26,
                                     color=colors[arm], alpha=.45, linewidths=0, zorder=3)
                mean = sum(values) / len(values)
                axis.scatter(x, mean, s=65, marker="D", color=colors[arm],
                             edgecolors="white", linewidths=.7, zorder=4,
                             label=phase + str(arm) if index == 0 else None)
        axis.axhline(.9, color="#98a2b3", linewidth=1, linestyle=(0, (4, 4)))
        axis.set_title(f"{phase}: {title}", loc="left", fontsize=11, pad=14)
        axis.set_xticks(range(4), labels)
        axis.set_xlim(-.55, 3.55)
        axis.set_ylim(-.03, 1.06)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", color="#eaecf0", linewidth=.8, zorder=0)
        axis.legend(loc="lower right", frameon=False, ncol=2)
    figure.suptitle("Native-language development at 2,400 updates", x=.06,
                    ha="left", fontsize=18, fontweight="bold")
    figure.text(.06, .915, "A: three fixed L1 seed pairs (dots); mean (diamonds).  B: one seed pair.  Dashed line: 90% target.",
                color="#475467", fontsize=10)
    figure.text(.06, .025, "Authored scene task; shared visual prior. Development scores do not establish general L1 fluency or language difficulty.",
                color="#475467", fontsize=9)
    figure.subplots_adjust(left=.07, right=.98, bottom=.1, top=.85, hspace=.38, wspace=.16)
    figure.savefig(directory / "paired-final.svg", metadata={"Date": None})
    # A raster preview is useful for checking labels locally; the SVG is the repository artifact.
    figure.savefig("/tmp/dovis-mastery-ab-preview.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
