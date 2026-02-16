# Cross-Lingual Embedding Similarity and Polysemy in Multilingual Models

Do words with similar meanings in different languages have similar embeddings? And what happens when words have multiple meanings?

## Key Findings

- **Translation equivalents have dramatically similar embeddings** in both mBERT and XLM-R: cosine similarity of 0.22–0.77 (after centering) vs. ~0.00 for random pairs (Cohen's d up to 3.0)
- **Polysemy degrades cross-lingual similarity**: Monosemous words have 15–65% higher similarity than highly polysemous words (5+ WordNet senses), with a consistent negative correlation (ρ = −0.07 to −0.37)
- **Context recovers alignment**: Same-sense cross-lingual pairs in context show significantly higher similarity than different-sense pairs (Cohen's d = 1.0–1.6), demonstrating that contextualized embeddings can overcome the polysemy problem
- **Middle-to-upper layers are best**: Layer 10 (of 12) provides optimal sense-discriminative cross-lingual alignment in both models
- **Language distance matters**: Romance languages (FR, ES) show the strongest alignment with English; Russian (Cyrillic script) shows the weakest

## How to Reproduce

```bash
# 1. Create and activate virtual environment
uv venv && source .venv/bin/activate

# 2. Install dependencies
uv pip install torch transformers numpy scipy matplotlib seaborn nltk scikit-learn tqdm pandas

# 3. Download NLTK data
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# 4. Run experiments (requires GPU, ~3.5 min on 2x RTX 3090)
python src/experiment.py

# 5. Run layer analysis for Experiment 3
python src/exp3_layer_analysis.py

# 6. Generate plots
python src/visualize.py
python src/visualize_exp3_layers.py
```

## File Structure

```
├── REPORT.md                  # Full research report with results
├── README.md                  # This file
├── planning.md                # Research plan and methodology
├── literature_review.md       # Literature review
├── resources.md               # Resource catalog
├── src/
│   ├── experiment.py          # Main experiment script (Exp 1-4)
│   ├── exp3_layer_analysis.py # Supplementary layer analysis
│   ├── visualize.py           # Main visualization script
│   └── visualize_exp3_layers.py # Layer analysis plots
├── results/
│   ├── all_results.json       # All experiment results
│   ├── exp3_layer_analysis.json
│   └── plots/                 # All generated figures
├── datasets/
│   ├── muse/                  # MUSE bilingual dictionaries
│   ├── semeval2017-task2/     # SemEval-2017 Task 2
│   └── mcl-wic/               # MCL-WiC dataset
├── papers/                    # Downloaded research papers
└── code/                      # Cloned baseline repositories
```

## Models Used

- `bert-base-multilingual-cased` (mBERT) — 178M params, 104 languages
- `xlm-roberta-base` (XLM-R) — 278M params, 100 languages

## Datasets

- **MUSE Bilingual Dictionaries** (Facebook Research) — 420K translation pairs
- **SemEval-2017 Task 2** — Cross-lingual word similarity benchmark
- **MCL-WiC** (SemEval-2021) — Cross-lingual word-in-context disambiguation

See [REPORT.md](REPORT.md) for the full analysis.
