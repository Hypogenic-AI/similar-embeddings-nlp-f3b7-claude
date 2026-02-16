# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project: "Do words with similar meanings in different languages have similar embeddings?" Resources include papers, datasets, and code repositories.

---

## Papers
Total papers downloaded: **19**

| # | Title | Authors | Year | File | Key Relevance |
|---|-------|---------|------|------|---------------|
| 1 | Survey of Cross-lingual Word Embedding Models | Ruder et al. | 2019 | `papers/ruder2017_survey_crosslingual_embeddings.pdf` | Foundation: CLWE taxonomy |
| 2 | XLM-R | Conneau et al. | 2020 | `papers/conneau2020_xlmr.pdf` | SOTA multilingual model |
| 3 | How Multilingual is Multilingual BERT? | Pires et al. | 2019 | `papers/pires2019_multilingual_bert.pdf` | mBERT cross-lingual abilities |
| 4 | Emerging Cross-lingual Structure | Wu & Conneau | 2020 | `papers/chi2020_emerging_crosslingual.pdf` | Universal embedding symmetries |
| 5 | BERTology Primer | Rogers et al. | 2020 | `papers/rogers2020_bertology.pdf` | BERT internal representations |
| 6 | Language Neutrality of Multilingual Reps | Libovicky et al. | 2020 | `papers/libovicky2020_language_neutrality.pdf` | Language-specific bias in mBERT |
| 7 | Language-Neutral mBERT? | Wu & Dredze | 2019 | `papers/wu2019_language_neutral_mbert.pdf` | Language neutrality probing |
| 8 | First Align, then Predict | Dufter & Schutze | 2021 | `papers/dufter2021_first_align_predict.pdf` | mBERT alignment mechanisms |
| 9 | Cross-lingual Alignment Methods | Cao et al. | 2020 | `papers/cao2020_alignment_methods.pdf` | Alignment method comparison |
| 10 | Unsupervised Multilingual Word Embeddings | Chen & Cardie | 2018 | `papers/chen2018_unsupervised_multilingual.pdf` | Unsupervised CLWE |
| 11 | Multi-sense Multilingual Embeddings | Upadhyay et al. | 2017 | `papers/upadhyay2017_multisense_multilingual.pdf` | Polysemy + multilingual signals |
| 12 | SensEmBERT | Scarlini et al. | 2020 | `papers/scarlini2020_sensembert.pdf` | Sense embeddings from BERT |
| 13 | Cross-lingual Multi-Sense Mapping | Zhang et al. | 2019 | `papers/zhang2019_crosslingual_multisense.pdf` | Polysemy hurts alignment |
| 14 | SemEval-2017 Task 2 | Camacho-Collados et al. | 2017 | `papers/camacho2017_semeval_multilingual_similarity.pdf` | Benchmark: word similarity |
| 15 | SemEval-2021 MCL-WiC | Martelli et al. | 2021 | `papers/martelli2021_semeval_mcl_wic.pdf` | Benchmark: WiC disambiguation |
| 16 | Multilingual Sentence-BERT | Reimers & Gurevych | 2020 | `papers/reimers2020_sentence_multilingual.pdf` | Multilingual sentence embeddings |
| 17 | Language-Agnostic Representations | Agic & Vulic | 2020 | `papers/agic2020_inducing_language_agnostic.pdf` | Language-agnostic methods |
| 18 | Universal Representations Across Languages | Chi et al. | 2020 | `papers/chi2020_universal_representations.pdf` | Universal cross-lingual reps |
| 19 | It's not Greek to mBERT | Gonen et al. | 2020 | `papers/gonen2020_its_not_greek.pdf` | Word translation from mBERT |

See `papers/README.md` for detailed descriptions.

---

## Datasets
Total datasets downloaded: **3**

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| MUSE Bilingual Dictionaries | Facebook Research | 420K word pairs across 5 language pairs | Bilingual Lexicon Induction | `datasets/muse/` | EN-FR, EN-DE, EN-ES, EN-RU, EN-ZH |
| SemEval-2017 Task 2 | SemEval | 500 pairs × 5 langs + 10 cross-lingual sets | Cross-lingual Word Similarity | `datasets/semeval2017-task2/` | EN, DE, ES, IT, FA |
| MCL-WiC | SapienzaNLP / SemEval-2021 | 8K train + dev/test in 5 langs | Word-in-Context Disambiguation | `datasets/mcl-wic/` | AR, ZH, EN, FR, RU |

See `datasets/README.md` for detailed descriptions and download instructions.

---

## Code Repositories
Total repositories cloned: **4**

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| MUSE | github.com/facebookresearch/MUSE | Cross-lingual embedding alignment (supervised & unsupervised) | `code/MUSE/` | Includes 110 bilingual dictionaries and evaluation tools |
| VecMap | github.com/artetxem/vecmap | Cross-lingual word embedding mappings | `code/vecmap/` | 4 alignment modes, evaluation tools |
| xling-eval | github.com/codogogo/xling-eval | Cross-lingual embedding evaluation | `code/xling-eval/` | 28 language pairs, multiple mapping methods |
| Wikipedia2Vec | github.com/wikipedia2vec/wikipedia2vec | Joint word + entity embeddings from Wikipedia | `code/wikipedia2vec/` | Pre-trained models in 12 languages |

See `code/README.md` for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. **Paper-finder service** (diligent mode) with three complementary queries:
   - "cross-lingual word embeddings similar meanings multilingual models"
   - "polysemy word sense disambiguation multilingual embeddings"
   - "multilingual BERT mBERT XLM-R cross-lingual representation similarity"
2. Results aggregated and ranked by relevance score and citation count
3. 181 unique papers identified; top 19 downloaded based on relevance (score >= 3) and direct applicability

### Selection Criteria
- **Papers**: Prioritized papers that (a) directly study cross-lingual embedding similarity, (b) analyze multilingual model representations, (c) address polysemy in cross-lingual settings, or (d) provide evaluation benchmarks
- **Datasets**: Selected benchmarks that enable measuring (a) translation equivalent similarity, (b) graded cross-lingual word similarity, and (c) sense-level cross-lingual disambiguation
- **Code**: Selected tools for (a) cross-lingual embedding alignment and evaluation, (b) bilingual lexicon induction, and (c) sense-aware embeddings

### Challenges Encountered
- Some arXiv IDs from paper-finder did not match the expected papers (ID collisions with papers from other fields); resolved by downloading from ACL Anthology instead
- The "It's not Greek to mBERT" paper downloaded from Findings of EMNLP was actually a different paper; the correct content was verified from the abstract

### Gaps and Workarounds
- **No dedicated polysemy-annotated cross-lingual dataset**: MCL-WiC partially fills this gap by testing sense disambiguation across languages. For polysemy analysis, we can use WordNet/BabelNet to identify polysemous words in MUSE dictionaries.
- **No pre-computed multilingual model embeddings**: These will need to be generated during the experiment phase using HuggingFace transformers.

---

## Recommendations for Experiment Design

Based on gathered resources, we recommend:

### 1. Primary Dataset(s)
- **MUSE bilingual dictionaries**: For measuring embedding similarity of translation equivalents in multilingual models. Large coverage (420K pairs) enables robust statistical analysis.
- **SemEval-2017 Task 2**: For evaluating correlation between model-computed similarity and human judgments across languages.
- **MCL-WiC**: For testing sense-level cross-lingual embedding similarity.

### 2. Baseline Methods
- **mBERT** (`bert-base-multilingual-cased`): Standard multilingual baseline
- **XLM-R** (`xlm-roberta-base`): State-of-the-art multilingual model
- **Aligned FastText** (via MUSE): Static embedding baseline for comparison

### 3. Evaluation Metrics
- **Cosine similarity** of translation pair embeddings (primary metric for the hypothesis)
- **Spearman correlation** with human similarity judgments (SemEval-2017)
- **Accuracy** on MCL-WiC cross-lingual task
- **Effect of polysemy**: Compare similarity for monosemous vs. polysemous translation pairs

### 4. Code to Adapt/Reuse
- **MUSE evaluation scripts**: For BLI evaluation and cross-lingual similarity computation
- **xling-eval**: For standardized cross-lingual embedding evaluation
- **HuggingFace transformers**: For extracting embeddings from mBERT and XLM-R
- **NLTK/spaCy WordNet interface**: For identifying polysemous words and their sense counts

### 5. Suggested Experimental Pipeline
```
1. Load multilingual models (mBERT, XLM-R) via HuggingFace
2. For each word pair in MUSE dictionaries:
   a. Extract embeddings (mean-pool subword tokens, try different layers)
   b. Compute cosine similarity
   c. Label as monosemous/polysemous using WordNet
3. Analyze: Do translation equivalents have high cosine similarity?
4. Compare: Is similarity lower for polysemous words?
5. Evaluate on SemEval-2017 Task 2 cross-lingual similarity
6. Evaluate on MCL-WiC cross-lingual disambiguation
7. Report results across language pairs and model layers
```
