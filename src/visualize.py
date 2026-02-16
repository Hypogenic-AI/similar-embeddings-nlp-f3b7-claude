"""
Visualization script for cross-lingual embedding similarity experiments.
Generates publication-quality plots from experiment results.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS = Path(__file__).parent.parent / "results"
PLOTS = RESULTS / "plots"
PLOTS.mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})
sns.set_style("whitegrid")


def load_results():
    with open(RESULTS / "all_results.json") as f:
        return json.load(f)


def plot_exp1_translation_vs_random(results):
    """Bar chart comparing translation pair similarity vs random pairs across languages and models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for idx, (model_key, model_label) in enumerate([("mbert", "mBERT"), ("xlmr", "XLM-R")]):
        ax = axes[idx]
        exp1 = results[model_key]["exp1_translation_similarity"]
        lang_pairs = sorted(exp1.keys())

        # Use last hidden layer
        layer = "-1"
        trans_means = []
        trans_stds = []
        rand_means = []
        rand_stds = []
        labels = []

        for lp in lang_pairs:
            if layer in exp1[lp] or int(layer) in exp1[lp]:
                key = layer if layer in exp1[lp] else int(layer)
                data = exp1[lp][key]
                trans_means.append(data["translation_sim_mean"])
                trans_stds.append(data["translation_sim_std"])
                rand_means.append(data["random_sim_mean"])
                rand_stds.append(data["random_sim_std"])
                labels.append(lp.upper())

        x = np.arange(len(labels))
        width = 0.35
        bars1 = ax.bar(x - width / 2, trans_means, width, yerr=trans_stds,
                        label='Translation pairs', color='#2196F3', alpha=0.85, capsize=3)
        bars2 = ax.bar(x + width / 2, rand_means, width, yerr=rand_stds,
                        label='Random pairs', color='#FF9800', alpha=0.85, capsize=3)

        ax.set_xlabel('Language Pair')
        ax.set_ylabel('Cosine Similarity (centered)')
        ax.set_title(f'{model_label}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(bottom=-0.1)

    fig.suptitle('Experiment 1: Translation Pair Similarity vs. Random Pairs', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS / "exp1_translation_vs_random.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS / 'exp1_translation_vs_random.png'}")


def plot_exp1_layer_analysis(results):
    """Line plot showing similarity across different layers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for idx, (model_key, model_label) in enumerate([("mbert", "mBERT"), ("xlmr", "XLM-R")]):
        ax = axes[idx]
        exp1 = results[model_key]["exp1_translation_similarity"]

        for lp in sorted(exp1.keys()):
            layers = sorted([int(k) if k != '-1' else -1 for k in exp1[lp].keys()])
            sims = []
            for l in layers:
                key = str(l)
                if key in exp1[lp]:
                    sims.append(exp1[lp][key]["translation_sim_mean"])
                elif l in exp1[lp]:
                    sims.append(exp1[lp][l]["translation_sim_mean"])
            # Map -1 to actual last layer number for display
            display_layers = [l if l >= 0 else 12 for l in layers]
            ax.plot(display_layers, sims, marker='o', label=lp.upper(), linewidth=2)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean Cosine Similarity (centered)')
        ax.set_title(f'{model_label}')
        ax.legend()

    fig.suptitle('Translation Pair Similarity Across Layers', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS / "exp1_layer_analysis.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS / 'exp1_layer_analysis.png'}")


def plot_exp2_monosemous_vs_polysemous(results):
    """Bar chart comparing similarity for monosemous vs polysemous words."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for idx, (model_key, model_label) in enumerate([("mbert", "mBERT"), ("xlmr", "XLM-R")]):
        ax = axes[idx]
        exp2 = results[model_key]["exp2_monosemous_vs_polysemous"]
        lang_pairs = sorted(exp2.keys())

        x = np.arange(len(lang_pairs))
        width = 0.25

        mono_means = [exp2[lp]["monosemous"]["sim_mean"] for lp in lang_pairs]
        mono_stds = [exp2[lp]["monosemous"]["sim_std"] for lp in lang_pairs]
        poly_means = [exp2[lp]["polysemous_2_4"]["sim_mean"] for lp in lang_pairs]
        poly_stds = [exp2[lp]["polysemous_2_4"]["sim_std"] for lp in lang_pairs]
        hpoly_means = []
        hpoly_stds = []
        has_hpoly = False
        for lp in lang_pairs:
            hp = exp2[lp]["highly_polysemous_5plus"]
            if hp["sim_mean"] is not None:
                hpoly_means.append(hp["sim_mean"])
                hpoly_stds.append(hp["sim_std"])
                has_hpoly = True
            else:
                hpoly_means.append(0)
                hpoly_stds.append(0)

        ax.bar(x - width, mono_means, width, yerr=mono_stds,
               label='Monosemous (1 sense)', color='#4CAF50', alpha=0.85, capsize=3)
        ax.bar(x, poly_means, width, yerr=poly_stds,
               label='Polysemous (2-4 senses)', color='#FF9800', alpha=0.85, capsize=3)
        if has_hpoly:
            ax.bar(x + width, hpoly_means, width, yerr=hpoly_stds,
                   label='Highly poly. (5+ senses)', color='#F44336', alpha=0.85, capsize=3)

        ax.set_xlabel('Language Pair')
        ax.set_ylabel('Cosine Similarity (centered)')
        ax.set_title(f'{model_label}')
        ax.set_xticks(x)
        ax.set_xticklabels([lp.upper() for lp in lang_pairs])
        ax.legend()

    fig.suptitle('Experiment 2: Monosemous vs. Polysemous Translation Pairs', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS / "exp2_mono_vs_poly.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS / 'exp2_mono_vs_poly.png'}")


def plot_exp2_sense_count_scatter(results):
    """Scatter/line plot: mean similarity vs. number of senses."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (model_key, model_label) in enumerate([("mbert", "mBERT"), ("xlmr", "XLM-R")]):
        ax = axes[idx]
        exp2 = results[model_key]["exp2_monosemous_vs_polysemous"]

        for lp in sorted(exp2.keys()):
            by_sense = exp2[lp].get("by_sense_count", {})
            if not by_sense:
                continue
            sense_counts = sorted([int(k) for k in by_sense.keys()])
            mean_sims = [by_sense[str(k)]["mean_sim"] for k in sense_counts]
            ax.plot(sense_counts, mean_sims, marker='o', label=lp.upper(), linewidth=2)

        ax.set_xlabel('Number of WordNet Senses')
        ax.set_ylabel('Mean Cosine Similarity')
        ax.set_title(f'{model_label}')
        ax.legend()

    fig.suptitle('Cross-Lingual Similarity by Number of Word Senses', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS / "exp2_sense_count_scatter.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS / 'exp2_sense_count_scatter.png'}")


def plot_exp3_same_vs_diff_sense(results):
    """Bar chart showing same-sense vs diff-sense contextualized similarity."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for idx, (model_key, model_label) in enumerate([("mbert", "mBERT"), ("xlmr", "XLM-R")]):
        ax = axes[idx]
        exp3 = results[model_key]["exp3_contextualized_sense_similarity"]
        lang_pairs = sorted(exp3.keys())

        x = np.arange(len(lang_pairs))
        width = 0.35

        same_means = [exp3[lp]["same_sense_sim_mean"] for lp in lang_pairs]
        same_stds = [exp3[lp]["same_sense_sim_std"] for lp in lang_pairs]
        diff_means = [exp3[lp]["diff_sense_sim_mean"] for lp in lang_pairs]
        diff_stds = [exp3[lp]["diff_sense_sim_std"] for lp in lang_pairs]

        ax.bar(x - width / 2, same_means, width, yerr=same_stds,
               label='Same sense (T)', color='#4CAF50', alpha=0.85, capsize=3)
        ax.bar(x + width / 2, diff_means, width, yerr=diff_stds,
               label='Different sense (F)', color='#F44336', alpha=0.85, capsize=3)

        # Annotate with Cohen's d
        for i, lp in enumerate(lang_pairs):
            d = exp3[lp]["cohens_d"]
            ax.annotate(f"d={d:.2f}", (i, max(same_means[i], diff_means[i]) + 0.05),
                       ha='center', fontsize=9, fontweight='bold')

        ax.set_xlabel('Language Pair')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title(f'{model_label}')
        ax.set_xticks(x)
        ax.set_xticklabels([lp.upper() for lp in lang_pairs])
        ax.legend()

    fig.suptitle('Experiment 3: Same-Sense vs. Different-Sense Cross-Lingual Similarity', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS / "exp3_same_vs_diff_sense.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS / 'exp3_same_vs_diff_sense.png'}")


def plot_exp4_semeval_correlations(results):
    """Bar chart of Spearman correlations on SemEval-2017 Task 2."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = ["mbert", "xlmr"]
    model_labels = ["mBERT", "XLM-R"]
    colors = ['#2196F3', '#4CAF50']

    all_pairs = set()
    for mk in models:
        all_pairs.update(results[mk]["exp4_semeval_crosslingual"].keys())
    lang_pairs = sorted(all_pairs)

    x = np.arange(len(lang_pairs))
    width = 0.35

    for i, (mk, ml) in enumerate(zip(models, model_labels)):
        exp4 = results[mk]["exp4_semeval_crosslingual"]
        correlations = [exp4.get(lp, {}).get("spearman_centered", 0) for lp in lang_pairs]
        ax.bar(x + i * width - width / 2, correlations, width, label=ml, color=colors[i], alpha=0.85)

    ax.set_xlabel('Language Pair')
    ax.set_ylabel('Spearman Correlation')
    ax.set_title('Experiment 4: Cross-Lingual Word Similarity (SemEval-2017 Task 2)')
    ax.set_xticks(x)
    ax.set_xticklabels([lp.upper() for lp in lang_pairs])
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOTS / "exp4_semeval_correlations.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS / 'exp4_semeval_correlations.png'}")


def plot_summary_table(results):
    """Create summary table as a figure."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Build table data
    headers = ['Experiment', 'Model', 'Key Metric', 'Value', 'Significance']
    rows = []

    for mk, ml in [("mbert", "mBERT"), ("xlmr", "XLM-R")]:
        # Exp 1
        exp1 = results[mk]["exp1_translation_similarity"]
        for lp in sorted(exp1.keys())[:2]:  # Top 2 language pairs
            key = "-1" if "-1" in exp1[lp] else -1
            d = exp1[lp][key]
            rows.append([f'Exp1: {lp}', ml, "Cohen's d (trans vs rand)",
                        f'{d["cohens_d"]:.3f}', f'p={d["p_value"]:.1e}'])

        # Exp 2
        exp2 = results[mk]["exp2_monosemous_vs_polysemous"]
        for lp in sorted(exp2.keys())[:2]:
            d = exp2[lp]
            rows.append([f'Exp2: {lp}', ml, "Cohen's d (mono vs poly)",
                        f'{d["mono_vs_poly_cohens_d"]:.3f}',
                        f'p={d["mono_vs_poly_pvalue"]:.1e}'])

        # Exp 3
        exp3 = results[mk]["exp3_contextualized_sense_similarity"]
        for lp in sorted(exp3.keys()):
            d = exp3[lp]
            rows.append([f'Exp3: {lp}', ml, "Cohen's d (same vs diff sense)",
                        f'{d["cohens_d"]:.3f}', f'p={d["p_value"]:.1e}'])

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color header
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#2196F3')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    plt.title('Summary of Statistical Results', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(PLOTS / "summary_table.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {PLOTS / 'summary_table.png'}")


def main():
    results = load_results()

    plot_exp1_translation_vs_random(results)
    plot_exp1_layer_analysis(results)
    plot_exp2_monosemous_vs_polysemous(results)
    plot_exp2_sense_count_scatter(results)
    plot_exp3_same_vs_diff_sense(results)
    plot_exp4_semeval_correlations(results)
    plot_summary_table(results)

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
