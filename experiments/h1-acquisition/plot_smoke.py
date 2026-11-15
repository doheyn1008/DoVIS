"""Plot recorded development metrics only; never loads or updates a learner."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.loads((args.run / "learning-curves.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True, layout="constrained")
    styles = {"aligned": ("#1565c0", "-", "o"), "no_update": ("#555555", "--", "s"),
              "shuffled": ("#e87912", "-.", "^"), "reset_understanding": ("#8e44ad", ":", "D")}
    for column, language in enumerate(("ko", "ja")):
        target = "ja" if language == "ko" else "ko"
        for row, metric in enumerate(("foreign_authored", "native_authored")):
            ax = axes[row, column]
            for condition, (color, linestyle, marker) in styles.items():
                points = sorted((p for p in data if p["learner"] == language and p["metric"] == metric
                                 and p["condition"] == condition), key=lambda p: p["exchange"])
                ax.plot([p["exchange"] for p in points], [p["raw_accuracy"] for p in points],
                        label=condition, color=color, linestyle=linestyle, marker=marker, linewidth=2, markersize=5)
            ax.axhline(.25, color="#999999", linewidth=1, linestyle="--", label="4-choice chance" if row == 0 else None)
            ax.set(ylim=(-.02, 1.02), xlim=(-.5, 32.5), xticks=[0, 8, 16, 32], yticks=[0, .25, .5, .75, 1])
            ax.grid(alpha=.18)
            ax.set_title(f"{language.upper()} learner: " + (f"{target.upper()} understanding" if row == 0 else "native-language retention"))
            if column == 0:
                ax.set_ylabel("Accuracy (448 authored queries)")
            if row == 1:
                ax.set_xlabel("Received peer utterances / shared PNG exposures")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("H1 development pilot: one KO/JA block, 32 exposures, fixed speakers\n"
                 "L2-unexposed development pairs; no protected test or language-distance inference", fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
