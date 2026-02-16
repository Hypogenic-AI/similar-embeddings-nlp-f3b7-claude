"""Plot Experiment 3 layer analysis showing Cohen's d across layers."""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS = Path(__file__).parent.parent / "results"
PLOTS = RESULTS / "plots"
plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})
sns.set_style("whitegrid")


def main():
    with open(RESULTS / "exp3_layer_analysis.json") as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for idx, (model_key, model_label) in enumerate([("mbert", "mBERT"), ("xlmr", "XLM-R")]):
        ax = axes[idx]
        for lp in sorted(results[model_key].keys()):
            layers = sorted(results[model_key][lp].keys(), key=lambda x: int(x))
            display_layers = [int(l) if int(l) >= 0 else 12 for l in layers]
            cohens_ds = [results[model_key][lp][l]["cohens_d"] for l in layers]
            ax.plot(display_layers, cohens_ds, marker='o', label=lp.upper(), linewidth=2, markersize=8)

        ax.set_xlabel('Layer')
        ax.set_ylabel("Cohen's d (same vs. different sense)")
        ax.set_title(f'{model_label}')
        ax.legend()
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')
        ax.annotate('Large effect (0.8)', xy=(1, 0.83), fontsize=9, color='gray')

    fig.suptitle('Experiment 3: Sense Discrimination Across Layers (with centering)', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS / "exp3_layer_cohens_d.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS / 'exp3_layer_cohens_d.png'}")

    # Also plot same vs diff similarity across layers
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (model_key, model_label) in enumerate([("mbert", "mBERT"), ("xlmr", "XLM-R")]):
        ax = axes[idx]
        colors = {'en-fr': '#2196F3', 'en-zh': '#4CAF50', 'en-ru': '#FF9800'}
        for lp in sorted(results[model_key].keys()):
            layers = sorted(results[model_key][lp].keys(), key=lambda x: int(x))
            display_layers = [int(l) if int(l) >= 0 else 12 for l in layers]
            same_means = [results[model_key][lp][l]["same_mean"] for l in layers]
            diff_means = [results[model_key][lp][l]["diff_mean"] for l in layers]
            c = colors.get(lp, 'gray')
            ax.plot(display_layers, same_means, marker='o', color=c, linewidth=2,
                   label=f'{lp.upper()} same', linestyle='-')
            ax.plot(display_layers, diff_means, marker='s', color=c, linewidth=2,
                   label=f'{lp.upper()} diff', linestyle='--')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean Cosine Similarity (centered)')
        ax.set_title(f'{model_label}')
        ax.legend(fontsize=8, ncol=2)

    fig.suptitle('Contextualized Sense Similarity Across Layers', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS / "exp3_layer_similarity.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS / 'exp3_layer_similarity.png'}")


if __name__ == "__main__":
    main()
